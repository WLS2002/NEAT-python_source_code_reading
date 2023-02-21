

# NEAT-python  源码阅读

## 代码结构

genes -> genome -> (species) -> population

## 代码解读

### config

在所有代码前，应该先看明白这个。因为后面所有代码都以config为参数。

#### config.py

```python
# 一个ConfigParameter，表示config中的一个值。
class ConfigParameter(object):
    """Contains information about one configuration item."""

    def __init__(self, name, value_type, default=None):
        self.name = name
        self.value_type = value_type
        self.default = default

    def __repr__(self):  # repr()将python里的对象，转化成可以被解释器读取的字符串。eval可以将这个字符串变回对象。
        if self.default is None:
            return f"ConfigParameter({self.name!r}, {self.value_type!r})"
        return f"ConfigParameter({self.name!r}, {self.value_type!r}, {self.default!r})"

    def parse(self, section, config_parser):
        if int == self.value_type:
            return config_parser.getint(section, self.name)
        if bool == self.value_type:
            return config_parser.getboolean(section, self.name)
        if float == self.value_type:
            return config_parser.getfloat(section, self.name)
        if list == self.value_type:
            v = config_parser.get(section, self.name)
            return v.split(" ")
        if str == self.value_type:
            return config_parser.get(section, self.name)

        raise RuntimeError(f"Unexpected configuration type: {self.value_type!r}")

    def interpret(self, config_dict):
        """
        Converts the config_parser output into the proper type,
        supplies defaults if available and needed, and checks for some errors.
        """
        value = config_dict.get(self.name)
        if value is None:
            if self.default is None:
                raise RuntimeError('Missing configuration item: ' + self.name)
            else:
                warnings.warn(f"Using default {self.default!r} for '{self.name!s}'", DeprecationWarning)
                if (str != self.value_type) and isinstance(self.default, self.value_type):
                    return self.default
                else:
                    value = self.default

        try:
            if str == self.value_type:
                return str(value)
            if int == self.value_type:
                return int(value)
            if bool == self.value_type:
                if value.lower() == "true":
                    return True
                elif value.lower() == "false":
                    return False
                else:
                    raise RuntimeError(self.name + " must be True or False")
            if float == self.value_type:
                return float(value)
            if list == self.value_type:
                return value.split(" ")
        except Exception:
            raise RuntimeError(
                f"Error interpreting config item '{self.name}' with value {value!r} and type {self.value_type}")

        raise RuntimeError("Unexpected configuration type: " + repr(self.value_type))

    def format(self, value):
        if list == self.value_type:
            return " ".join(value)
        return str(value)
```

这里有一个有意思的函数。

```python
def write_pretty_params(f, config, params):
    param_names = [p.name for p in params]
    longest_name = max(len(name) for name in param_names)
    param_names.sort()  # 先排序
    params = dict((p.name, p) for p in params)

    for name in param_names:
        p = params[name]
        f.write(f'{p.name.ljust(longest_name)} = {p.format(getattr(config, p.name))}\n')  # 等号得对齐
```

config来了

```python
class Config(object):
    """A container for user-configurable parameters of NEAT."""

    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str),
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('reset_on_extinction', bool),
                ConfigParameter('no_fitness_termination', bool, False)]
    '''
	xor里面的config创建方式：
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    '''
    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename, config_information=None):
        # Check that the provided types have the required methods.
        assert hasattr(genome_type, 'parse_config')
        assert hasattr(reproduction_type, 'parse_config')
        assert hasattr(species_set_type, 'parse_config')
        assert hasattr(stagnation_type, 'parse_config')

        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type
        self.config_information = config_information

        if not os.path.isfile(filename):
            raise Exception('No such config file: ' + os.path.abspath(filename))

        parameters = ConfigParser()
        with open(filename) as f:
            parameters.read_file(f)  # 功能比较强大

        # NEAT configuration
        if not parameters.has_section('NEAT'):
            raise RuntimeError("'NEAT' section not found in NEAT configuration file.")

        param_list_names = []
        for p in self.__params:  # 从section里获取parameters
            if p.default is None:
                setattr(self, p.name, p.parse('NEAT', parameters))
            else:
                try:
                    setattr(self, p.name, p.parse('NEAT', parameters))
                except Exception:
                    setattr(self, p.name, p.default)
                    warnings.warn(f"Using default {p.default!r} for '{p.name!s}'",
                                  DeprecationWarning)
            param_list_names.append(p.name)
        param_dict = dict(parameters.items('NEAT'))
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown (section 'NEAT') configuration items:\n" + "\n\t".join(unknown_list))
            raise UnknownConfigItemError(f"Unknown (section 'NEAT') configuration item {unknown_list[0]!s}")

        # Parse type sections.
        # 解析其他section的config
        genome_dict = dict(parameters.items(genome_type.__name__))
        self.genome_config = genome_type.parse_config(genome_dict)

        species_set_dict = dict(parameters.items(species_set_type.__name__))
        self.species_set_config = species_set_type.parse_config(species_set_dict)

        stagnation_dict = dict(parameters.items(stagnation_type.__name__))
        self.stagnation_config = stagnation_type.parse_config(stagnation_dict)

        reproduction_dict = dict(parameters.items(reproduction_type.__name__))
        self.reproduction_config = reproduction_type.parse_config(reproduction_dict)
		
    def save(self, filename):  # 保存当前config到文件里
        with open(filename, 'w') as f:
            f.write('# The `NEAT` section specifies parameters particular to the NEAT algorithm\n')
            f.write('# or the experiment itself.  This is the only required section.\n')
            f.write('[NEAT]\n')
            write_pretty_params(f, self, self.__params)

            f.write(f'\n[{self.genome_type.__name__}]\n')
            self.genome_type.write_config(f, self.genome_config)

            f.write(f'\n[{self.species_set_type.__name__}]\n')
            self.species_set_type.write_config(f, self.species_set_config)

            f.write(f'\n[{self.stagnation_type.__name__}]\n')
            self.stagnation_type.write_config(f, self.stagnation_config)

            f.write(f'\n[{self.reproduction_type.__name__}]\n')
            self.reproduction_type.write_config(f, self.reproduction_config)
```

### gene

gene指基因，多个基因构成基因组。

#### gene.py

```python
class BaseGene(object):
    """
    Handles functions shared by multiple types of genes (both node and connection),
    including crossover and calling mutation methods.
    """

    def __init__(self, key):  # 创建base gene只需要一个key
        self.key = key

    def __str__(self):  # tostring方法，不细看了
        attrib = ['key'] + [a.name for a in self._gene_attributes]
        attrib = [f'{a}={getattr(self, a)}' for a in attrib]
        return f'{self.__class__.__name__}({", ".join(attrib)})'

    def __lt__(self, other):  # 通过key比较大小
        assert isinstance(self.key, type(other.key)), f"Cannot compare keys {self.key!r} and {other.key!r}"
        return self.key < other.key

    @classmethod
    def parse_config(cls, config, param_dict):  # 解析config？
        pass

    @classmethod
    def get_config_params(cls):  # 有点逆天，就是一个简单的获取_gene_attributes的代码，写出了让人完全看不懂的感觉。
        params = []
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(cls, '__gene_attributes__'))
            warnings.warn(
                f"Class '{cls.__name__!s}' {cls!r} needs '_gene_attributes' not '__gene_attributes__'",
                DeprecationWarning)
        for a in cls._gene_attributes:  # 仔细看了下，就干了这点事
            params += a.get_config_params()
        return params

    @classmethod
    def validate_attributes(cls, config):  # 检验config是否合法
        for a in cls._gene_attributes:
            a.validate(config)

    def init_attributes(self, config):  # initialize，其实就是做了代替了这样的代码：self.bias = ..... 把_gene_attributes里的属性都这么写了。从config里读出了这些数据的初始值
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config))

    def mutate(self, config):  # 出现了，单基因mutate！
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def copy(self):  # copy
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))

        return new_gene

    def crossover(self, gene2):  # 两个基因之间crossover！
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        new_gene = self.__class__(self.key)  # new_gene用的居然是self.key吗
        for a in self._gene_attributes:
            if random() > 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))

        return new_gene
    '''
    这段代码再告诉我，_gene_attributes非常重要！那么它到底是啥？
    '''
```



```python
class DefaultNodeGene(BaseGene):
    # 重要的_gene_attributes终于来了！认真看看
    _gene_attributes = [FloatAttribute('bias'),
                        FloatAttribute('response'),
                        StringAttribute('activation', options=''),  # 节点的激活函数，如sigmoid，relu
                        StringAttribute('aggregation', options='')] # 本层节点接受上层节点值的方式, 如mean, sum

    def __init__(self, key):
        assert isinstance(key, int), f"DefaultNodeGene key must be an int, not {key!r}"
        # Node Gene的key是一个int
        BaseGene.__init__(self, key)

    def distance(self, other, config):  # 获取和另外一个gene的距离？有点奇怪
        d = abs(self.bias - other.bias) + abs(self.response - other.response)
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient


# TODO: Do an ablation study to determine whether the enabled setting is
# important--presumably mutations that set the weight to near zero could
# provide a similar effect depending on the weight range, mutation rate,
# and aggregation function. (Most obviously, a near-zero weight for the
# `product` aggregation function is rather more important than one giving
# an output of 1 from the connection, for instance!)
class DefaultConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute('weight'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), f"DefaultConnectionGene key must be a tuple, not {key!r}"
        # 注意，这里的key是一个tuple，表示(input_node_key, output_node_key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient
    
'''
一共有两种gene，NodeGene和ConnectionGene，这和论文里的说法是一致的。
NodeGene的attributes：
	bias：好理解
	response：目前还不懂
	activation：处理非线性的激活函数？
	aggregation：本层节点接受上层节点值的方式, 如mean, sum
ConnectionGene：
	weight：你懂的
	enabled：表示连接是否启用
'''
```

![image-20230215111146995](C:\Users\wls\AppData\Roaming\Typora\typora-user-images\image-20230215111146995.png)

这里附上论文里的一张图。

#### attribute.py

attribute指基因里的值。Node Gene有4个attribute，而Connection Gene有2个。

```python
class BaseAttribute(object):  # 感觉一个小小的attribute不用写的这么复杂呀。
	'''
	对于所有使用到的FloatAttribute，BoolAttribute, **default_dict都没有;
	StringAttribute，**default_dict有一个options=''
	只是用了上面三种Attribute。
	'''
    
    """Superclass for the type-specialized attribute subclasses, used by genes."""

    def __init__(self, name, **default_dict):  # name
        self.name = name
        # TODO: implement a mechanism that allows us to detect and report unused configuration items.
        for n, default in default_dict.items():
            self._config_items[n] = [self._config_items[n][0], default]
        for n in self._config_items:
            setattr(self, n + "_name", self.config_item_name(n))

    def config_item_name(self, config_item_base_name):
        return f"{self.name}_{config_item_base_name}"

    def get_config_params(self):
        return [ConfigParameter(self.config_item_name(n), ci[0], ci[1])
                for n, ci in self._config_items.items()]
        
class FloatAttribute(BaseAttribute): 
    """
    Class for floating-point numeric attributes,
    such as the response of a node or the weight of a connection.
    """
    
    '''
    初始值是一个随机值，由init_mean, init_stdev, init_type产生
    比如init_type是gaussian，那么初始值就是self.clamp(gauss(mean, stdev))
    '''
    _config_items = {"init_mean": [float, None],     # 产生初始值
                     "init_stdev": [float, None],    # 产生初始值
                     "init_type": [str, 'gaussian'], # 产生初始值
                     "replace_rate": [float, None],  # 突变时重新生成初始值的概率
                     "mutate_rate": [float, None],   # 突变率
                     "mutate_power": [float, None],  # 突变方差，突变 = ori_value + gauss(0.0, mutate_power)
                     "max_value": [float, None],
                     "min_value": [float, None]}

    def clamp(self, value, config):  #该简单的地方写的复杂
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        return max(min(value, max_value), min_value)  # 6

    def init_value(self, config):   # 产生初始值
        mean = getattr(config, self.init_mean_name)  # mean = config.init_mean_name
        stdev = getattr(config, self.init_stdev_name)
        init_type = getattr(config, self.init_type_name).lower()

        if ('gauss' in init_type) or ('normal' in init_type):
            return self.clamp(gauss(mean, stdev), config)

        if 'uniform' in init_type:
            min_value = max(getattr(config, self.min_value_name),
                            (mean - (2 * stdev)))
            max_value = min(getattr(config, self.max_value_name),
                            (mean + (2 * stdev)))
            return uniform(min_value, max_value)

        raise RuntimeError(f"Unknown init_type {getattr(config, self.init_type_name)!r} for {self.init_type_name!s}")

    def mutate_value(self, value, config):  
        # 由原有值突变出一个新的值。根据replace_rate and mutate_rate. 决定新值的产生方式。
        # r = rand(); (0, mutate_rate) -> 扰动； [mutate_rate, mutate_rate + replace_rate) -> new value; otherwise no change.
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        mutate_rate = getattr(config, self.mutate_rate_name)

        r = random()
        if r < mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name)
            return self.clamp(value + gauss(0.0, mutate_power), config)

        replace_rate = getattr(config, self.replace_rate_name)

        if r < replace_rate + mutate_rate:
            return self.init_value(config)

        return value

    def validate(self, config):  # 验证config是否正确
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        if max_value < min_value:
            raise RuntimeError("Invalid min/max configuration for {self.name}")
```

后面还有IntAttribute和BoolAttribute，这边不放上来了。基本和FloatAttribute一样。

```python
class StringAttribute(BaseAttribute):  # 可选值是options，一个list。突变就是从里面随机选一个新的。
    """
    Class for string attributes such as the aggregation function of a node,
    which are selected from a list of options.
    """
    _config_items = {"default": [str, 'random'],
                     "options": [list, None],
                     "mutate_rate": [float, None]}

    def init_value(self, config):
        default = getattr(config, self.default_name)

        if default.lower() in ('none', 'random'):
            options = getattr(config, self.options_name)
            return choice(options)

        return default

    def mutate_value(self, value, config):
        mutate_rate = getattr(config, self.mutate_rate_name)

        if mutate_rate > 0:
            r = random()
            if r < mutate_rate:
                options = getattr(config, self.options_name)
                return choice(options)

        return value

    def validate(self, config):
        default = getattr(config, self.default_name)
        if default not in ('none', 'random'):
            options = getattr(config, self.options_name)
            if default not in options:
                raise RuntimeError(f'Invalid initial value {default} for {self.name}')
            assert default in options
```

### genome

geneme指基因组。表示population中的每个个体。

#### genome.py

先看看genome的config

```python
# config这部分看着真头晕，暂且跳过一下
class DefaultGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('num_hidden', int),
                        ConfigParameter('feed_forward', bool),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('conn_delete_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('single_structural_mutation', bool, 'false'),
                        ConfigParameter('structural_mutation_surer', str, 'default'),
                        ConfigParameter('initial_connection', str, 'unconnected')]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        self.node_gene_type.validate_attributes(self)
        self.connection_gene_type.validate_attributes(self)

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ['1', 'yes', 'true', 'on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0', 'no', 'false', 'off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)

        self.node_indexer = None

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")
            f.write(f'initial_connection      = {self.initial_connection} {self.connection_fraction}\n')
        else:
            f.write(f'initial_connection      = {self.initial_connection}\n')

        assert self.initial_connection in self.allowed_connectivity

        write_pretty_params(f, self, [p for p in self._params
                                      if 'initial_connection' not in p.name])

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            if node_dict:
                self.node_indexer = count(max(list(node_dict)) + 1)
            else:
                self.node_indexer = count(max(list(node_dict)) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)
```

还是来看看真正的genome代码

```python
class DefaultGenome(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene  # 就是前面Gene里面看的两种Gene。Node和Connection
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}

        # Fitness results.
        self.fitness = None
```

创建新genome的代码

```python
    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""

        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        # Add hidden nodes if requested.
        if config.num_hidden > 0:
            for i in range(config.num_hidden):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                node = self.create_node(config, node_key)
                self.nodes[node_key] = node

        # Add connections based on initial connectivity type.

        if 'fs_neat' in config.initial_connection:  # 这个好像是新的论文
            if config.initial_connection == 'fs_neat_nohidden':
                self.connect_fs_neat_nohidden(config)
            elif config.initial_connection == 'fs_neat_hidden':
                self.connect_fs_neat_hidden(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = fs_neat will not connect to hidden nodes;",
                        "\tif this is desired, set initial_connection = fs_neat_nohidden;",
                        "\tif not, set initial_connection = fs_neat_hidden",
                        sep='\n', file=sys.stderr)
                self.connect_fs_neat_nohidden(config)
        elif 'full' in config.initial_connection:
            if config.initial_connection == 'full_nodirect':
                self.connect_full_nodirect(config)
            elif config.initial_connection == 'full_direct':
                self.connect_full_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = full with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = full_nodirect;",
                        "\tif not, set initial_connection = full_direct",
                        sep='\n', file=sys.stderr)
                self.connect_full_nodirect(config)
        elif 'partial' in config.initial_connection:
            if config.initial_connection == 'partial_nodirect':
                self.connect_partial_nodirect(config)
            elif config.initial_connection == 'partial_direct':
                self.connect_partial_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = partial with hidden nodes will not do direct input-output connections;",
                        f"\tif this is desired, set initial_connection = partial_nodirect {config.connection_fraction};",
                        f"\tif not, set initial_connection = partial_direct {config.connection_fraction}",
                        sep='\n', file=sys.stderr)
                self.connect_partial_nodirect(config)
```

看起来创建新Genome的步骤为：

1. 创建输出层隐藏节点
2. 创建初始的hidden nodes
3. 根据config，创建不同种类的初始connection

上面有很多种conneciton，代码在这里：

```python
    def connect_fs_neat_nohidden(self, config):  # 随机选一个input和所有output连起来
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        input_id = choice(config.input_keys)
        for output_id in config.output_keys:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection
            
    def connect_fs_neat_hidden(self, config):  # 随机选一个input和所有output以及hidden连起来
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        input_id = choice(config.input_keys)
        others = [i for i in self.nodes if i not in config.input_keys]
        for output_id in others:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection
            

    def compute_full_connections(self, config, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        hidden = [i for i in self.nodes if i not in config.output_keys]
        output = [i for i in self.nodes if i in config.output_keys]  # ?output keys又不会被删掉。应该一直在nodes里面吧。
        connections = []
        if hidden:  # 有hidden，input连hidden，hidden连output
            for input_id in config.input_keys:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):  # 没有hidden或者direct=True，input连output
            for input_id in config.input_keys:
                for output_id in output:
                    connections.append((input_id, output_id))

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:  # 考虑到有循环神经网络，这里还需要自连
            for i in self.nodes:
                connections.append((i, i))

        return connections

    def connect_full_nodirect(self, config):  # 在compute_full_connections的基础上全连, 不会直接input连output
        """
        Create a fully-connected genome
        (except without direct input-output unless no hidden nodes).
        """
        for input_id, output_id in self.compute_full_connections(config, False):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_full_direct(self, config):  # 在compute_full_connections的基础上全连, 会直接input连output。
        """ Create a fully-connected genome, including direct input-output connections. """
        for input_id, output_id in self.compute_full_connections(config, True):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_nodirect(self, config):  # 在compute_full_connections的基础上选择部分连, 不会直接input连output。 
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, False)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_direct(self, config):  # 在compute_full_connections的基础上选择部分连, 会直接input连output
        """
        Create a partially-connected genome,
        including (possibly) direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, True)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection
```



两个genome之间的crossover：

```python
    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)
```

传入两个genome parent，改变自身的值？



再看看mutate:

```python
    def mutate(self, config):
        """ Mutates this genome. """

        if config.single_structural_mutation:
            div = max(1, (config.node_add_prob + config.node_delete_prob +
                          config.conn_add_prob + config.conn_delete_prob))
            r = random()
            if r < (config.node_add_prob / div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob) / div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob) / div):
                self.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob) / div):
                self.mutate_delete_connection()
        else:
            if random() < config.node_add_prob:
                self.mutate_add_node(config)

            if random() < config.node_delete_prob:
                self.mutate_delete_node(config)

            if random() < config.conn_add_prob:
                self.mutate_add_connection(config)

            if random() < config.conn_delete_prob:
                self.mutate_delete_connection()

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)
```

直接mutate自己。

结构突变：

​	可能会有4种情况，mutate_add_node，mutate_delete_node，mutate_add_connection和mutate_delete_connection；

​	如果single_structural_mutation = True，那么上述四种情况，至多只会发生一次；

​	如果single_structural_mutation = False，上述四种情况的概率会分别考虑，有可能四种情况都会发生。总共进行4次突变。

权重突变：

​	直接变

下面是Mutate Structure不同情况的代码：

```python
	def mutate_add_node(self, config):
        if not self.connections:  # 如果一个connection都没有的话，就不能从connection里面变出node了。
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)  # 变成加conneciton
            return

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False  # disable了

        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id, 1.0, True)  # 前面的权值就是1
        self.add_connection(config, new_node_id, o, conn_to_split.weight, True)  # 后面的是本来的connection的权值
    
    def add_connection(self, config, input_key, output_key, weight, enabled):
        # TODO: Add further validation of this connection addition?
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = enabled
        self.connections[key] = connection

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        possible_outputs = list(self.nodes)  # nodes里面本来就没有input node
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            # TODO: Should this be using mutation to/from rates? Hairy to configure...
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(self.connections), key):  # 避免回路。保证拓扑性
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):  # 随便找一个node删了。不过不能删output node。
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in self.nodes if k not in config.output_keys]  # output nodes不能删
        if not available_nodes:
            return -1

        del_key = choice(available_nodes)

        connections_to_delete = set()
        for k, v in self.connections.items():
            if del_key in v.key:  # 似乎和del_key in k等价？总之就是删掉和这个node有关的connections
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key

    def mutate_delete_connection(self):  # 随便找一个connection删了，6。我好奇为啥不是disable掉。而是直接给删了。怎么会是呢
        if self.connections:
            key = choice(list(self.connections.keys()))
            del self.connections[key]
```

mutate就这样。



下面是论文里提到的genome distance。这个要用在后面物种那里。

```python
    def distance(self, other, config):  # distance = node_distance + connection_distance
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:  # 至少有一个人有一个node
            disjoint_nodes = 0
            for k2 in other.nodes:
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)  # 就算有这个node，也要考虑node里面参数的distance

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance +
                             (config.compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in self.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance
```



还有一个小方法。在后面统计里可能会用到：

```python
    def size(self):  # 就是统计node和enable connections的数量。可以用于衡量网络的复杂程度。
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), num_enabled_connections
```

上面出现过的create_node和create_conneciton

```python
    @staticmethod
    def create_node(config, node_id):
        node = config.node_gene_type(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = config.connection_gene_type((input_id, output_id))
        connection.init_attributes(config)
        return connection
```



最后，还有一个用来copy的方法。估计没啥，不细看了。

```python
    def get_pruned_copy(self, genome_config):
        used_node_genes, used_connection_genes = get_pruned_genes(self.nodes, self.connections,
                                                                  genome_config.input_keys, genome_config.output_keys)
        new_genome = DefaultGenome(None)
        new_genome.nodes = used_node_genes
        new_genome.connections = used_connection_genes
        return new_genome


def get_pruned_genes(node_genes, connection_genes, input_keys, output_keys):
    used_nodes = required_for_output(input_keys, output_keys, connection_genes)
    used_pins = used_nodes.union(input_keys)

    # Copy used nodes into a new genome.
    used_node_genes = {}
    for n in used_nodes:
        used_node_genes[n] = copy.deepcopy(node_genes[n])

    # Copy enabled and used connections into the new genome.
    used_connection_genes = {}
    for key, cg in connection_genes.items():
        in_node_id, out_node_id = key
        if cg.enabled and in_node_id in used_pins and out_node_id in used_pins:
            used_connection_genes[key] = copy.deepcopy(cg)

    return used_node_genes, used_connection_genes
```

今天就到这。

### stagnation

本来在看reproduction的，突然发现这段代码完全没看。晕

```python
class DefaultStagnation(DefaultClassConfig):
    """Keeps track of whether species are making progress and helps remove ones that are not."""

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('species_fitness_func', str, 'mean'),
                                   ConfigParameter('max_stagnation', int, 15),
                                   ConfigParameter('species_elitism', int, 0)])

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.stagnation_config = config

        # min, max, mean, median之类的
        self.species_fitness_func = stat_functions.get(config.species_fitness_func)
        
        if self.species_fitness_func is None:
            raise RuntimeError(
                "Unexpected species fitness func: {0!r}".format(config.species_fitness_func))

        self.reporters = reporters

    def update(self, species_set, generation):
        """
        Required interface method. Updates species fitness history information,
        checking for ones that have not improved in max_stagnation generations,
        and - unless it would result in the number of species dropping below the configured
        species_elitism parameter if they were removed,
        in which case the highest-fitness species are spared -
        returns a list with stagnant species marked for removal.
        """
        species_data = []
        for sid, s in species_set.species.items():
            if s.fitness_history:
                prev_fitness = max(s.fitness_history)
            else:
                prev_fitness = -sys.float_info.max

            s.fitness = self.species_fitness_func(s.get_fitnesses())  # 从物种里每个个体的分数计算出物种的分数
            s.fitness_history.append(s.fitness)
            s.adjusted_fitness = None
            if prev_fitness is None or s.fitness > prev_fitness:
                s.last_improved = generation

            species_data.append((sid, s))

        # Sort in ascending fitness order.
        species_data.sort(key=lambda x: x[1].fitness)  # 根据物种的fitness将(sid, s)排序

        result = []
        species_fitnesses = []  # 好像完全没有用，笑了
        num_non_stagnant = len(species_data)
        for idx, (sid, s) in enumerate(species_data):
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            stagnant_time = generation - s.last_improved  # 停滞时间
            is_stagnant = False
            if num_non_stagnant > self.stagnation_config.species_elitism:  # 物种数量有上限限制？
                is_stagnant = stagnant_time >= self.stagnation_config.max_stagnation  # 停滞太长的就滚

            if (len(species_data) - idx) <= self.stagnation_config.species_elitism:  # 后排分高的一定不会停滞
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((sid, s, is_stagnant))
            species_fitnesses.append(s.fitness)

        return result
```

### reproduiction

用于不断生成新的genome

2023/2/20 

昨晚4点才睡的，现在是上午10点，精神状态有点差

希望上午能把这段看完。下午再看population，看完就全看完了。

```python
class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 1)])

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)  # 基因的key
        self.stagnation = stagnation  # 停滯？
        self.ancestors = {}

    def create_new(self, genome_type, genome_config, num_genomes):  # 还有其他的genome_type吗，好像就一个default呀
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()  # 第一批人 没有祖先，合理的

        return new_genomes
	
    # 这段看的有点迷惑，也不知道论文在哪里这么说了
    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        # 计算出新一代每一个物种里的个体数量
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)  # adjusted_fitness所占的比重。同时控制下界
            else:
                s = min_species_size
			
            d = (s - ps) * 0.5  # 种群大小渐进变化，而不是一步变到s
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1
			# d == 0，则不变是吧
            
            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn  # 还需要归一化，控制pop_size
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts

    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        all_fitnesses = []
        remaining_species = []
        
        # stagnant同来判断物种是否停滞。分数很多代没有增长就是停滞
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}  # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in afs.members.values()])  # 这里是直接取mean了
            af = (msf - min_fitness) / fitness_range # 压缩到0-1之间？
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)  # type: float
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)
        
        # 获取新的物种大小
        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)  # 从大到小排序

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:  # 保留前几个分高的
                    new_population[i] = m
                    spawn -= 1
			
            if spawn <= 0: 
            continue
    		# 上面的代码是先把精英保留下来，直接放到下一代
            
            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
                                         len(old_members)))  # 只需要分比较高的人作为父母繁衍
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]
			
            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)  # 完全随机选，跟分数无关，甚至不考虑会不会重复
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).  # 父母就是可以一样。生一个一样的孩子
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)  # 创建一个child对象，然后调用它的crossover方法
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)  # 新人要突变
                # TODO: if config.genome_config.feed_forward, no cycles should exist
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)  # 记录下祖先，虽然不知道这个祖先有啥用

        return new_population

```

### species

物种，NEAT中的核心概念。前面已经有了判断两个基因组距离的方法，这里从不同距离来划分物种。

"Divides the population into species based on genomic distances."

关于这一部分，论文里提到的划分物种的方法是这样的：

![image-20230217103630822](C:\Users\wls\AppData\Roaming\Typora\typora-user-images\image-20230217103630822.png)

希望能找到对应的代码

```python
class Species(object):
    def __init__(self, key, generation):  # generation是一个int，表示物种是在哪一代创建的
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative = None  # 物种的代表genome
        self.members = {}           # 成员
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []

    def update(self, representative, members):
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        return [m.fitness for m in self.members.values()]
    
# 一个物种，species不可数（好奇怪）
```

```python
class GenomeDistanceCache(object):
    def __init__(self, config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0, genome1):
        g0 = genome0.key
        g1 = genome1.key
        d = self.distances.get((g0, g1))
        if d is None:
            # Distance is not already computed.
            d = genome0.distance(genome1, self.config)
            self.distances[g0, g1] = d
            self.distances[g1, g0] = d
            self.misses += 1
        else:
            self.hits += 1

        return d
# Cache
```

直接看代码

```python
class DefaultSpeciesSet(DefaultClassConfig):  # 好多个species
    """ Encapsulates the default speciation scheme. """

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = {}  # 刚开始没有物种，合理的
        self.genome_to_species = {}

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float)])

    def speciate(self, config, population, generation):  # population: dict(key -> genome)，generation: int
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(population, dict)

        compatibility_threshold = self.species_set_config.compatibility_threshold

        # Find the best representatives for each existing species.
        unspeciated = set(population)
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        for sid, s in self.species.items():
            candidates = []
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)
            
		# 每一代的物种里，都有一个representative genome用来代表这个物种。上一代的representative genome可能已经被演化整不一样了
        # 所以要从这一带所有人里面找一个和那个人最像的，来作为这一带该物种的representative genome。
       
        # Partition population into species based on genetic similarity.
        while unspeciated:
            gid = unspeciated.pop()
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in new_representatives.items():
                rep = population[rid]
                d = distances(rep, g)
                if d < compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]
                
       	# 这里和论文里有些不一样。论文里说的是个体直接被划分到第一个距离小于阈值的物种。而这里是划分到了小于阈值里距离最小的物种。
        # 如果没有小于阈值的物种，则创建新物种。
        # 如此看来，当最开始一个物种都没有的时候，第一个个体直接形成了第一个物种。然后后面的个体就开始判定。

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in new_representatives.items():
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)  # generation是一个int，表示物种是在哪一代创建的
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        # Mean and std genetic distance info report
        if len(population) > 1:
            gdmean = mean(distances.distances.values())
            gdstdev = stdev(distances.distances.values())
            self.reporters.info(
                'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev))

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
```

2023.2.17 上午：

看完了这段代码，做一个总结：

1. 初始状态下，物种数量为0；
2. 第一次调用speciate时，第一个个体直接成为第一个物种，并且作为这个物种的representative。后面的个体接着判定。
3. 后续的每一次调用speciate，完成以下事情：
   1. 为每个物种重新选定representative。因为上一代的representative genome可能已经被演化整不一样了，所以要从这一带所有人里面找一个和那个人最像的，来作为这一带该物种的representative genome。
   2. 把其他的个体划分到物种里。这里和论文里有些不一样。论文里说的是个体直接被划分到第一个距离小于阈值的物种。而这里是划分到了小于阈值里距离最小的物种。如果没有小于阈值的物种，则创建新物种。

整体来说不是很难。今天上午状态挺好的，看的比较投入。

### population

代码阅读的最后一部分！！！！！看了45天了都

```python
class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        #  fitness评价标准 
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def run(self, fitness_function, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(self.population.items()), self.config)

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break
                    
			# 上面都是分数统计。记录最高分、判断是否停止
            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)
            # reproduce产生下一代

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)
			# speciate维护新的种群
            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome
```

### graph

最后，关于NEAT网络计算，代码里有一些图论算法。

```python
def creates_cycle(connections, test):
    """
    Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False
```

判断已有无环图connections里，如果加入了一个新的连接test($i \to o$)，是否会产生环。

算法的逻辑很简单，就是从o出发，bfs搜索是否有有一条路径能到达i，如果有，那么有环。



```python
def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a set of identifiers of required nodes.
    """
    assert not set(inputs).intersection(outputs)

    required = set(outputs)
    s = set(outputs)
    while 1:
        # Find nodes not in s whose output is consumed by a node in s.
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:  # 如果没有新加入的node，break
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:  # 如果新加入的node全是input，也要break
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required
```

判断图里哪些node在计算过程中需要用到，逻辑也不难，就是从后往前倒推就行。



```python
def feed_forward_layers(inputs, outputs, connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """

    required = required_for_output(inputs, outputs, connections)

    layers = []
    s = set(inputs)
    while 1:
        # Find candidate nodes c for the next layer.  These nodes should connect
        # a node in s to a node not in s.
        c = set(b for (a, b) in connections if a in s and b not in s)  # 先找出这一层哪些节点有可能能算出来
        # Keep only the used nodes whose entire input set is contained in s.
        t = set()
        for n in c:  # not required的节点不用算，还需要判断这些点是否真的能算出来
            if n in required and all(a in s for (a, b) in connections if b == n):
                t.add(n)

        if not t:  # 没有新的点，break
            break

        layers.append(t)  # add这一层
        s = s.union(t)

    return layers
```

找出图的计算顺序，返回的结果是 一层层的node

### nn.feed_forward

差点忘了，还有神经网络计算的部分。如何将由Node和Connection两种基因构成的基因组，转化成可以用来做前向计算的网络。

```python
class FeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)  # 原来response是这个用处的...

        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        # 在graph.py里的方法。用处是返回一层层的计算节点。
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))  # Connection的Attribute--weight

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)  # 聚合方式
                activation_function = config.genome_config.activation_defs.get(ng.activation)  # 激活函数
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)

```



## 总结

2023/2/21 

花了大概3个工作日的时间，终于把原版NEAT的代码看完了。现在对它的运行结构已经有了比较细致地了解。在此做一个总结。

NEAT和我以前接触的演化算法不一样，在于两点。

1. 之前了解的演化算法都是在优化已有模型的参数，NEAT同时优化已有模型的参数和拓扑结构。
2. NEAT有一套物种保护机制，他将整个种群划分为不同的物种，在物种的内部进行淘汰、交叉、突变。

下面细说这两点：

### NeuralEvolution for Augmenting Topologies

每一个个体的基因组(Genome)由两类基因构成。Node Gene和Connection Gene。如它们的名字一样，Node Gene表示网络拓扑结构中的计算单元(Node)，Connection Gene表示两个Node之间的连接(Connection)。

Node Gene有以下Attributes:

1. bias: float, 偏置
2. response: float, 相应比重
3. activation: str, 激活函数 
4. aggregation: str, 聚合函数

Connection Gene有一下Attributes:

1. weight: float, 权重
2. enable: bool, 是否启用

网络计算方式:

 ```python
 for node, act_func, agg_func, bias, response, links in self.node_evals:
     node_inputs = []
     for i, w in links:
         node_inputs.append(self.values[i] * w)  # w -> weight 
         s = agg_func(node_inputs)  # 聚合函数，如sum，mean，square_sum
         self.values[node] = act_func(bias + response * s)
 ```

