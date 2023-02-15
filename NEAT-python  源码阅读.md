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

