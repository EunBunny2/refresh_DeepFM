??	
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02unknown8??
[
wVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namew
T
w/Read/ReadVariableOpReadVariableOpw*
_output_shapes	
:?*
dtype0
^
VVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameV
W
V/Read/ReadVariableOpReadVariableOpV*
_output_shapes

:*
dtype0
?
deep_fm/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*%
shared_namedeep_fm/dense/kernel
~
(deep_fm/dense/kernel/Read/ReadVariableOpReadVariableOpdeep_fm/dense/kernel*
_output_shapes
:	?@*
dtype0
|
deep_fm/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namedeep_fm/dense/bias
u
&deep_fm/dense/bias/Read/ReadVariableOpReadVariableOpdeep_fm/dense/bias*
_output_shapes
:@*
dtype0
?
deep_fm/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_namedeep_fm/dense_1/kernel
?
*deep_fm/dense_1/kernel/Read/ReadVariableOpReadVariableOpdeep_fm/dense_1/kernel*
_output_shapes

:@*
dtype0
?
deep_fm/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namedeep_fm/dense_1/bias
y
(deep_fm/dense_1/bias/Read/ReadVariableOpReadVariableOpdeep_fm/dense_1/bias*
_output_shapes
:*
dtype0
?
deep_fm/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_namedeep_fm/dense_2/kernel
?
*deep_fm/dense_2/kernel/Read/ReadVariableOpReadVariableOpdeep_fm/dense_2/kernel*
_output_shapes

:*
dtype0
?
deep_fm/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namedeep_fm/dense_2/bias
y
(deep_fm/dense_2/bias/Read/ReadVariableOpReadVariableOpdeep_fm/dense_2/bias*
_output_shapes
:*
dtype0
?
deep_fm/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_namedeep_fm/dense_3/kernel
?
*deep_fm/dense_3/kernel/Read/ReadVariableOpReadVariableOpdeep_fm/dense_3/kernel*
_output_shapes

:*
dtype0
?
deep_fm/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namedeep_fm/dense_3/bias
y
(deep_fm/dense_3/bias/Read/ReadVariableOpReadVariableOpdeep_fm/dense_3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
field_index
fm_layer
layers1
dropout1
layers2
dropout2
layers3
	final
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
 
q
field_index
w
V
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
 bias
!trainable_variables
"	variables
#regularization_losses
$	keras_api
R
%trainable_variables
&	variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+trainable_variables
,	variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
F
0
1
2
3
4
 5
)6
*7
/8
09
 
F
0
1
2
3
4
 5
)6
*7
/8
09
?
	trainable_variables
5layer_regularization_losses

regularization_losses
6layer_metrics
7non_trainable_variables
8metrics
	variables

9layers
 
 
<:
VARIABLE_VALUEw%fm_layer/w/.ATTRIBUTES/VARIABLE_VALUE
<:
VARIABLE_VALUEV%fm_layer/V/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
	variables
:layer_regularization_losses
regularization_losses
;layer_metrics
<non_trainable_variables
=metrics

>layers
SQ
VARIABLE_VALUEdeep_fm/dense/kernel)layers1/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdeep_fm/dense/bias'layers1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
	variables
?layer_regularization_losses
regularization_losses
@layer_metrics
Anon_trainable_variables
Bmetrics

Clayers
 
 
 
?
trainable_variables
	variables
Dlayer_regularization_losses
regularization_losses
Elayer_metrics
Fnon_trainable_variables
Gmetrics

Hlayers
US
VARIABLE_VALUEdeep_fm/dense_1/kernel)layers2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEdeep_fm/dense_1/bias'layers2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
?
!trainable_variables
"	variables
Ilayer_regularization_losses
#regularization_losses
Jlayer_metrics
Knon_trainable_variables
Lmetrics

Mlayers
 
 
 
?
%trainable_variables
&	variables
Nlayer_regularization_losses
'regularization_losses
Olayer_metrics
Pnon_trainable_variables
Qmetrics

Rlayers
US
VARIABLE_VALUEdeep_fm/dense_2/kernel)layers3/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEdeep_fm/dense_2/bias'layers3/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
?
+trainable_variables
,	variables
Slayer_regularization_losses
-regularization_losses
Tlayer_metrics
Unon_trainable_variables
Vmetrics

Wlayers
SQ
VARIABLE_VALUEdeep_fm/dense_3/kernel'final/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdeep_fm/dense_3/bias%final/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
?
1trainable_variables
2	variables
Xlayer_regularization_losses
3regularization_losses
Ylayer_metrics
Znon_trainable_variables
[metrics

\layers
 
 
 
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Vwdeep_fm/dense/kerneldeep_fm/dense/biasdeep_fm/dense_1/kerneldeep_fm/dense_1/biasdeep_fm/dense_2/kerneldeep_fm/dense_2/biasdeep_fm/dense_3/kerneldeep_fm/dense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_40196
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamew/Read/ReadVariableOpV/Read/ReadVariableOp(deep_fm/dense/kernel/Read/ReadVariableOp&deep_fm/dense/bias/Read/ReadVariableOp*deep_fm/dense_1/kernel/Read/ReadVariableOp(deep_fm/dense_1/bias/Read/ReadVariableOp*deep_fm/dense_2/kernel/Read/ReadVariableOp(deep_fm/dense_2/bias/Read/ReadVariableOp*deep_fm/dense_3/kernel/Read/ReadVariableOp(deep_fm/dense_3/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_40836
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamewVdeep_fm/dense/kerneldeep_fm/dense/biasdeep_fm/dense_1/kerneldeep_fm/dense_1/biasdeep_fm/dense_2/kerneldeep_fm/dense_2/biasdeep_fm/dense_3/kerneldeep_fm/dense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_40876΀
?
b
)__inference_dropout_1_layer_call_fn_40743

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_399142
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_2_layer_call_fn_40763

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_398232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_40691

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_397862
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_dense_2_layer_call_and_return_conditional_losses_39823

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_40738

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_398102
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_dense_3_layer_call_and_return_conditional_losses_40774

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_39786

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_40196
input_1
unknown:
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_397142
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_39914

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_dense_2_layer_call_and_return_conditional_losses_40754

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_dense_layer_call_and_return_conditional_losses_40660

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
B__inference_deep_fm_layer_call_and_return_conditional_losses_40037

inputs 
fm_layer_40002:
fm_layer_40004:	?
dense_40010:	?@
dense_40012:@
dense_1_40016:@
dense_1_40018:
dense_2_40022:
dense_2_40024:
dense_3_40029:
dense_3_40031:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall? fm_layer/StatefulPartitionedCall?
 fm_layer/StatefulPartitionedCallStatefulPartitionedCallinputsfm_layer_40002fm_layer_40004*
Tin
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes-
+:?????????:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_fm_layer_layer_call_and_return_conditional_losses_397552"
 fm_layer/StatefulPartitionedCallo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Reshape/shape?
ReshapeReshape)fm_layer/StatefulPartitionedCall:output:1Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_40010dense_40012*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_397752
dense/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_399472!
dropout/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_40016dense_1_40018*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_397992!
dense_1/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_399142#
!dropout_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_40022dense_2_40024*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_398232!
dense_2/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2)fm_layer/StatefulPartitionedCall:output:0(dense_2/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
dense_3/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_3_40029dense_3_40031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_398422!
dense_3/StatefulPartitionedCallu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shape?
	Reshape_1Reshape(dense_3/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1i
IdentityIdentityReshape_1:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall!^fm_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2D
 fm_layer/StatefulPartitionedCall fm_layer/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
C__inference_fm_layer_layer_call_and_return_conditional_losses_40638

inputs(
embedding_lookup_40610:,
mul_1_readvariableop_resource:	?
identity

identity_1??Mul_1/ReadVariableOp?embedding_lookups
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????>     2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshape?

embedding_lookup/idsConst*
_output_shapes	
:?*
dtype0*?

value?
B?
?"?	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                2
embedding_lookup/ids?
embedding_lookupResourceGatherembedding_lookup_40610embedding_lookup/ids:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/40610*
_output_shapes
:	?*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/40610*
_output_shapes
:	?2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	?2
embedding_lookup/Identity_1?
MulMulReshape:output:0$embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
Mul?
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Mul_1/ReadVariableOpn
Mul_1MulMul_1/ReadVariableOp:value:0inputs*
T0*(
_output_shapes
:??????????2
Mul_1p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesj
SumSum	Mul_1:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum?
Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_1/reduction_indicesn
Sum_1SumMul:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_1X
SquareSquareSum_1:output:0*
T0*#
_output_shapes
:?????????2
Square^
Square_1SquareMul:z:0*
T0*,
_output_shapes
:??????????2

Square_1?
Sum_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_2/reduction_indicess
Sum_2SumSquare_1:y:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_2[
SubSub
Square:y:0Sum_2:output:0*
T0*#
_output_shapes
:?????????2
SubW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_2/x^
mul_2Mulmul_2/x:output:0Sub:z:0*
T0*#
_output_shapes
:?????????2
mul_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_1/shape{
	Reshape_1ReshapeSum:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_2/shapex
	Reshape_2Reshape	mul_2:z:0Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_2\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1IdentityMul:z:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity_1x
NoOpNoOp^Mul_1/ReadVariableOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?(
?
B__inference_deep_fm_layer_call_and_return_conditional_losses_39851

inputs 
fm_layer_39756:
fm_layer_39758:	?
dense_39776:	?@
dense_39778:@
dense_1_39800:@
dense_1_39802:
dense_2_39824:
dense_2_39826:
dense_3_39843:
dense_3_39845:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall? fm_layer/StatefulPartitionedCall?
 fm_layer/StatefulPartitionedCallStatefulPartitionedCallinputsfm_layer_39756fm_layer_39758*
Tin
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes-
+:?????????:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_fm_layer_layer_call_and_return_conditional_losses_397552"
 fm_layer/StatefulPartitionedCallo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Reshape/shape?
ReshapeReshape)fm_layer/StatefulPartitionedCall:output:1Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_39776dense_39778*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_397752
dense/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_397862
dropout/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_39800dense_1_39802*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_397992!
dense_1/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_398102
dropout_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_39824dense_2_39826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_398232!
dense_2/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2)fm_layer/StatefulPartitionedCall:output:0(dense_2/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
dense_3/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_3_39843dense_3_39845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_398422!
dense_3/StatefulPartitionedCallu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shape?
	Reshape_1Reshape(dense_3/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1i
IdentityIdentityReshape_1:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall!^fm_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2D
 fm_layer/StatefulPartitionedCall fm_layer/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
'__inference_deep_fm_layer_call_fn_40525
input_1
unknown:
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_deep_fm_layer_call_and_return_conditional_losses_398512
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_40733

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_40721

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_deep_fm_layer_call_fn_40600
input_1
unknown:
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_deep_fm_layer_call_and_return_conditional_losses_400372
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_39947

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?[
?
B__inference_deep_fm_layer_call_and_return_conditional_losses_40265

inputs1
fm_layer_embedding_lookup_40202:5
&fm_layer_mul_1_readvariableop_resource:	?7
$dense_matmul_readvariableop_resource:	?@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?fm_layer/Mul_1/ReadVariableOp?fm_layer/embedding_lookup?
fm_layer/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????>     2
fm_layer/Reshape/shape?
fm_layer/ReshapeReshapeinputsfm_layer/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
fm_layer/Reshape?
fm_layer/embedding_lookup/idsConst*
_output_shapes	
:?*
dtype0*?

value?
B?
?"?	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                2
fm_layer/embedding_lookup/ids?
fm_layer/embedding_lookupResourceGatherfm_layer_embedding_lookup_40202&fm_layer/embedding_lookup/ids:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*2
_class(
&$loc:@fm_layer/embedding_lookup/40202*
_output_shapes
:	?*
dtype02
fm_layer/embedding_lookup?
"fm_layer/embedding_lookup/IdentityIdentity"fm_layer/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@fm_layer/embedding_lookup/40202*
_output_shapes
:	?2$
"fm_layer/embedding_lookup/Identity?
$fm_layer/embedding_lookup/Identity_1Identity+fm_layer/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	?2&
$fm_layer/embedding_lookup/Identity_1?
fm_layer/MulMulfm_layer/Reshape:output:0-fm_layer/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
fm_layer/Mul?
fm_layer/Mul_1/ReadVariableOpReadVariableOp&fm_layer_mul_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
fm_layer/Mul_1/ReadVariableOp?
fm_layer/Mul_1Mul%fm_layer/Mul_1/ReadVariableOp:value:0inputs*
T0*(
_output_shapes
:??????????2
fm_layer/Mul_1?
fm_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
fm_layer/Sum/reduction_indices?
fm_layer/SumSumfm_layer/Mul_1:z:0'fm_layer/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Sum?
 fm_layer/Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2"
 fm_layer/Sum_1/reduction_indices?
fm_layer/Sum_1Sumfm_layer/Mul:z:0)fm_layer/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Sum_1s
fm_layer/SquareSquarefm_layer/Sum_1:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Squarey
fm_layer/Square_1Squarefm_layer/Mul:z:0*
T0*,
_output_shapes
:??????????2
fm_layer/Square_1?
 fm_layer/Sum_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2"
 fm_layer/Sum_2/reduction_indices?
fm_layer/Sum_2Sumfm_layer/Square_1:y:0)fm_layer/Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Sum_2
fm_layer/SubSubfm_layer/Square:y:0fm_layer/Sum_2:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Subi
fm_layer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
fm_layer/mul_2/x?
fm_layer/mul_2Mulfm_layer/mul_2/x:output:0fm_layer/Sub:z:0*
T0*#
_output_shapes
:?????????2
fm_layer/mul_2?
fm_layer/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
fm_layer/Reshape_1/shape?
fm_layer/Reshape_1Reshapefm_layer/Sum:output:0!fm_layer/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
fm_layer/Reshape_1?
fm_layer/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
fm_layer/Reshape_2/shape?
fm_layer/Reshape_2Reshapefm_layer/mul_2:z:0!fm_layer/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
fm_layer/Reshape_2n
fm_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
fm_layer/concat/axis?
fm_layer/concatConcatV2fm_layer/Reshape_1:output:0fm_layer/Reshape_2:output:0fm_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
fm_layer/concato
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Reshape/shapez
ReshapeReshapefm_layer/Mul:z:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relu|
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Relu?
dropout_1/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
dropout_1/Identity?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Relu\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2fm_layer/concat:output:0dense_2/Relu:activations:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulconcat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shape~
	Reshape_1Reshapedense_3/Sigmoid:y:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1i
IdentityIdentityReshape_1:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^fm_layer/Mul_1/ReadVariableOp^fm_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2>
fm_layer/Mul_1/ReadVariableOpfm_layer/Mul_1/ReadVariableOp26
fm_layer/embedding_lookupfm_layer/embedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
(__inference_fm_layer_layer_call_fn_40649

inputs
unknown:
	unknown_0:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes-
+:?????????:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_fm_layer_layer_call_and_return_conditional_losses_397552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*,
_output_shapes
:??????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_40686

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_39799

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_40716

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_397992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_40707

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?m
?
B__inference_deep_fm_layer_call_and_return_conditional_losses_40500
input_11
fm_layer_embedding_lookup_40423:5
&fm_layer_mul_1_readvariableop_resource:	?7
$dense_matmul_readvariableop_resource:	?@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?fm_layer/Mul_1/ReadVariableOp?fm_layer/embedding_lookup?
fm_layer/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????>     2
fm_layer/Reshape/shape?
fm_layer/ReshapeReshapeinput_1fm_layer/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
fm_layer/Reshape?
fm_layer/embedding_lookup/idsConst*
_output_shapes	
:?*
dtype0*?

value?
B?
?"?	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                2
fm_layer/embedding_lookup/ids?
fm_layer/embedding_lookupResourceGatherfm_layer_embedding_lookup_40423&fm_layer/embedding_lookup/ids:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*2
_class(
&$loc:@fm_layer/embedding_lookup/40423*
_output_shapes
:	?*
dtype02
fm_layer/embedding_lookup?
"fm_layer/embedding_lookup/IdentityIdentity"fm_layer/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@fm_layer/embedding_lookup/40423*
_output_shapes
:	?2$
"fm_layer/embedding_lookup/Identity?
$fm_layer/embedding_lookup/Identity_1Identity+fm_layer/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	?2&
$fm_layer/embedding_lookup/Identity_1?
fm_layer/MulMulfm_layer/Reshape:output:0-fm_layer/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
fm_layer/Mul?
fm_layer/Mul_1/ReadVariableOpReadVariableOp&fm_layer_mul_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
fm_layer/Mul_1/ReadVariableOp?
fm_layer/Mul_1Mul%fm_layer/Mul_1/ReadVariableOp:value:0input_1*
T0*(
_output_shapes
:??????????2
fm_layer/Mul_1?
fm_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
fm_layer/Sum/reduction_indices?
fm_layer/SumSumfm_layer/Mul_1:z:0'fm_layer/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Sum?
 fm_layer/Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2"
 fm_layer/Sum_1/reduction_indices?
fm_layer/Sum_1Sumfm_layer/Mul:z:0)fm_layer/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Sum_1s
fm_layer/SquareSquarefm_layer/Sum_1:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Squarey
fm_layer/Square_1Squarefm_layer/Mul:z:0*
T0*,
_output_shapes
:??????????2
fm_layer/Square_1?
 fm_layer/Sum_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2"
 fm_layer/Sum_2/reduction_indices?
fm_layer/Sum_2Sumfm_layer/Square_1:y:0)fm_layer/Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Sum_2
fm_layer/SubSubfm_layer/Square:y:0fm_layer/Sum_2:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Subi
fm_layer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
fm_layer/mul_2/x?
fm_layer/mul_2Mulfm_layer/mul_2/x:output:0fm_layer/Sub:z:0*
T0*#
_output_shapes
:?????????2
fm_layer/mul_2?
fm_layer/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
fm_layer/Reshape_1/shape?
fm_layer/Reshape_1Reshapefm_layer/Sum:output:0!fm_layer/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
fm_layer/Reshape_1?
fm_layer/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
fm_layer/Reshape_2/shape?
fm_layer/Reshape_2Reshapefm_layer/mul_2:z:0!fm_layer/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
fm_layer/Reshape_2n
fm_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
fm_layer/concat/axis?
fm_layer/concatConcatV2fm_layer/Reshape_1:output:0fm_layer/Reshape_2:output:0fm_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
fm_layer/concato
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Reshape/shapez
ReshapeReshapefm_layer/Mul:z:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMuldense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_1/dropout/Mul|
dropout_1/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_1/dropout/Mul_1?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Relu\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2fm_layer/concat:output:0dense_2/Relu:activations:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulconcat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shape~
	Reshape_1Reshapedense_3/Sigmoid:y:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1i
IdentityIdentityReshape_1:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^fm_layer/Mul_1/ReadVariableOp^fm_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2>
fm_layer/Mul_1/ReadVariableOpfm_layer/Mul_1/ReadVariableOp26
fm_layer/embedding_lookupfm_layer/embedding_lookup:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?!
?
__inference__traced_save_40836
file_prefix 
savev2_w_read_readvariableop 
savev2_v_read_readvariableop3
/savev2_deep_fm_dense_kernel_read_readvariableop1
-savev2_deep_fm_dense_bias_read_readvariableop5
1savev2_deep_fm_dense_1_kernel_read_readvariableop3
/savev2_deep_fm_dense_1_bias_read_readvariableop5
1savev2_deep_fm_dense_2_kernel_read_readvariableop3
/savev2_deep_fm_dense_2_bias_read_readvariableop5
1savev2_deep_fm_dense_3_kernel_read_readvariableop3
/savev2_deep_fm_dense_3_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%fm_layer/w/.ATTRIBUTES/VARIABLE_VALUEB%fm_layer/V/.ATTRIBUTES/VARIABLE_VALUEB)layers1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layers1/bias/.ATTRIBUTES/VARIABLE_VALUEB)layers2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layers2/bias/.ATTRIBUTES/VARIABLE_VALUEB)layers3/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layers3/bias/.ATTRIBUTES/VARIABLE_VALUEB'final/kernel/.ATTRIBUTES/VARIABLE_VALUEB%final/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_w_read_readvariableopsavev2_v_read_readvariableop/savev2_deep_fm_dense_kernel_read_readvariableop-savev2_deep_fm_dense_bias_read_readvariableop1savev2_deep_fm_dense_1_kernel_read_readvariableop/savev2_deep_fm_dense_1_bias_read_readvariableop1savev2_deep_fm_dense_2_kernel_read_readvariableop/savev2_deep_fm_dense_2_bias_read_readvariableop1savev2_deep_fm_dense_3_kernel_read_readvariableop/savev2_deep_fm_dense_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*i
_input_shapesX
V: :?::	?@:@:@:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:?:$ 

_output_shapes

::%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: 
?.
?
!__inference__traced_restore_40876
file_prefix!
assignvariableop_w:	?&
assignvariableop_1_v::
'assignvariableop_2_deep_fm_dense_kernel:	?@3
%assignvariableop_3_deep_fm_dense_bias:@;
)assignvariableop_4_deep_fm_dense_1_kernel:@5
'assignvariableop_5_deep_fm_dense_1_bias:;
)assignvariableop_6_deep_fm_dense_2_kernel:5
'assignvariableop_7_deep_fm_dense_2_bias:;
)assignvariableop_8_deep_fm_dense_3_kernel:5
'assignvariableop_9_deep_fm_dense_3_bias:
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%fm_layer/w/.ATTRIBUTES/VARIABLE_VALUEB%fm_layer/V/.ATTRIBUTES/VARIABLE_VALUEB)layers1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layers1/bias/.ATTRIBUTES/VARIABLE_VALUEB)layers2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layers2/bias/.ATTRIBUTES/VARIABLE_VALUEB)layers3/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layers3/bias/.ATTRIBUTES/VARIABLE_VALUEB'final/kernel/.ATTRIBUTES/VARIABLE_VALUEB%final/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_wIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_vIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp'assignvariableop_2_deep_fm_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp%assignvariableop_3_deep_fm_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp)assignvariableop_4_deep_fm_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp'assignvariableop_5_deep_fm_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp)assignvariableop_6_deep_fm_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp'assignvariableop_7_deep_fm_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp)assignvariableop_8_deep_fm_dense_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp'assignvariableop_9_deep_fm_dense_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10f
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_11?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?i
?
 __inference__wrapped_model_39714
input_19
'deep_fm_fm_layer_embedding_lookup_39651:=
.deep_fm_fm_layer_mul_1_readvariableop_resource:	??
,deep_fm_dense_matmul_readvariableop_resource:	?@;
-deep_fm_dense_biasadd_readvariableop_resource:@@
.deep_fm_dense_1_matmul_readvariableop_resource:@=
/deep_fm_dense_1_biasadd_readvariableop_resource:@
.deep_fm_dense_2_matmul_readvariableop_resource:=
/deep_fm_dense_2_biasadd_readvariableop_resource:@
.deep_fm_dense_3_matmul_readvariableop_resource:=
/deep_fm_dense_3_biasadd_readvariableop_resource:
identity??$deep_fm/dense/BiasAdd/ReadVariableOp?#deep_fm/dense/MatMul/ReadVariableOp?&deep_fm/dense_1/BiasAdd/ReadVariableOp?%deep_fm/dense_1/MatMul/ReadVariableOp?&deep_fm/dense_2/BiasAdd/ReadVariableOp?%deep_fm/dense_2/MatMul/ReadVariableOp?&deep_fm/dense_3/BiasAdd/ReadVariableOp?%deep_fm/dense_3/MatMul/ReadVariableOp?%deep_fm/fm_layer/Mul_1/ReadVariableOp?!deep_fm/fm_layer/embedding_lookup?
deep_fm/fm_layer/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????>     2 
deep_fm/fm_layer/Reshape/shape?
deep_fm/fm_layer/ReshapeReshapeinput_1'deep_fm/fm_layer/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
deep_fm/fm_layer/Reshape?
%deep_fm/fm_layer/embedding_lookup/idsConst*
_output_shapes	
:?*
dtype0*?

value?
B?
?"?	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                2'
%deep_fm/fm_layer/embedding_lookup/ids?
!deep_fm/fm_layer/embedding_lookupResourceGather'deep_fm_fm_layer_embedding_lookup_39651.deep_fm/fm_layer/embedding_lookup/ids:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@deep_fm/fm_layer/embedding_lookup/39651*
_output_shapes
:	?*
dtype02#
!deep_fm/fm_layer/embedding_lookup?
*deep_fm/fm_layer/embedding_lookup/IdentityIdentity*deep_fm/fm_layer/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@deep_fm/fm_layer/embedding_lookup/39651*
_output_shapes
:	?2,
*deep_fm/fm_layer/embedding_lookup/Identity?
,deep_fm/fm_layer/embedding_lookup/Identity_1Identity3deep_fm/fm_layer/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	?2.
,deep_fm/fm_layer/embedding_lookup/Identity_1?
deep_fm/fm_layer/MulMul!deep_fm/fm_layer/Reshape:output:05deep_fm/fm_layer/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
deep_fm/fm_layer/Mul?
%deep_fm/fm_layer/Mul_1/ReadVariableOpReadVariableOp.deep_fm_fm_layer_mul_1_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%deep_fm/fm_layer/Mul_1/ReadVariableOp?
deep_fm/fm_layer/Mul_1Mul-deep_fm/fm_layer/Mul_1/ReadVariableOp:value:0input_1*
T0*(
_output_shapes
:??????????2
deep_fm/fm_layer/Mul_1?
&deep_fm/fm_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2(
&deep_fm/fm_layer/Sum/reduction_indices?
deep_fm/fm_layer/SumSumdeep_fm/fm_layer/Mul_1:z:0/deep_fm/fm_layer/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
deep_fm/fm_layer/Sum?
(deep_fm/fm_layer/Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2*
(deep_fm/fm_layer/Sum_1/reduction_indices?
deep_fm/fm_layer/Sum_1Sumdeep_fm/fm_layer/Mul:z:01deep_fm/fm_layer/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
deep_fm/fm_layer/Sum_1?
deep_fm/fm_layer/SquareSquaredeep_fm/fm_layer/Sum_1:output:0*
T0*#
_output_shapes
:?????????2
deep_fm/fm_layer/Square?
deep_fm/fm_layer/Square_1Squaredeep_fm/fm_layer/Mul:z:0*
T0*,
_output_shapes
:??????????2
deep_fm/fm_layer/Square_1?
(deep_fm/fm_layer/Sum_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2*
(deep_fm/fm_layer/Sum_2/reduction_indices?
deep_fm/fm_layer/Sum_2Sumdeep_fm/fm_layer/Square_1:y:01deep_fm/fm_layer/Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
deep_fm/fm_layer/Sum_2?
deep_fm/fm_layer/SubSubdeep_fm/fm_layer/Square:y:0deep_fm/fm_layer/Sum_2:output:0*
T0*#
_output_shapes
:?????????2
deep_fm/fm_layer/Suby
deep_fm/fm_layer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
deep_fm/fm_layer/mul_2/x?
deep_fm/fm_layer/mul_2Mul!deep_fm/fm_layer/mul_2/x:output:0deep_fm/fm_layer/Sub:z:0*
T0*#
_output_shapes
:?????????2
deep_fm/fm_layer/mul_2?
 deep_fm/fm_layer/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2"
 deep_fm/fm_layer/Reshape_1/shape?
deep_fm/fm_layer/Reshape_1Reshapedeep_fm/fm_layer/Sum:output:0)deep_fm/fm_layer/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
deep_fm/fm_layer/Reshape_1?
 deep_fm/fm_layer/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2"
 deep_fm/fm_layer/Reshape_2/shape?
deep_fm/fm_layer/Reshape_2Reshapedeep_fm/fm_layer/mul_2:z:0)deep_fm/fm_layer/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
deep_fm/fm_layer/Reshape_2~
deep_fm/fm_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
deep_fm/fm_layer/concat/axis?
deep_fm/fm_layer/concatConcatV2#deep_fm/fm_layer/Reshape_1:output:0#deep_fm/fm_layer/Reshape_2:output:0%deep_fm/fm_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
deep_fm/fm_layer/concat
deep_fm/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
deep_fm/Reshape/shape?
deep_fm/ReshapeReshapedeep_fm/fm_layer/Mul:z:0deep_fm/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2
deep_fm/Reshape?
#deep_fm/dense/MatMul/ReadVariableOpReadVariableOp,deep_fm_dense_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02%
#deep_fm/dense/MatMul/ReadVariableOp?
deep_fm/dense/MatMulMatMuldeep_fm/Reshape:output:0+deep_fm/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
deep_fm/dense/MatMul?
$deep_fm/dense/BiasAdd/ReadVariableOpReadVariableOp-deep_fm_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$deep_fm/dense/BiasAdd/ReadVariableOp?
deep_fm/dense/BiasAddBiasAdddeep_fm/dense/MatMul:product:0,deep_fm/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
deep_fm/dense/BiasAdd?
deep_fm/dense/ReluReludeep_fm/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
deep_fm/dense/Relu?
deep_fm/dropout/IdentityIdentity deep_fm/dense/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
deep_fm/dropout/Identity?
%deep_fm/dense_1/MatMul/ReadVariableOpReadVariableOp.deep_fm_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%deep_fm/dense_1/MatMul/ReadVariableOp?
deep_fm/dense_1/MatMulMatMul!deep_fm/dropout/Identity:output:0-deep_fm/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
deep_fm/dense_1/MatMul?
&deep_fm/dense_1/BiasAdd/ReadVariableOpReadVariableOp/deep_fm_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&deep_fm/dense_1/BiasAdd/ReadVariableOp?
deep_fm/dense_1/BiasAddBiasAdd deep_fm/dense_1/MatMul:product:0.deep_fm/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
deep_fm/dense_1/BiasAdd?
deep_fm/dense_1/ReluRelu deep_fm/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
deep_fm/dense_1/Relu?
deep_fm/dropout_1/IdentityIdentity"deep_fm/dense_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
deep_fm/dropout_1/Identity?
%deep_fm/dense_2/MatMul/ReadVariableOpReadVariableOp.deep_fm_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%deep_fm/dense_2/MatMul/ReadVariableOp?
deep_fm/dense_2/MatMulMatMul#deep_fm/dropout_1/Identity:output:0-deep_fm/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
deep_fm/dense_2/MatMul?
&deep_fm/dense_2/BiasAdd/ReadVariableOpReadVariableOp/deep_fm_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&deep_fm/dense_2/BiasAdd/ReadVariableOp?
deep_fm/dense_2/BiasAddBiasAdd deep_fm/dense_2/MatMul:product:0.deep_fm/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
deep_fm/dense_2/BiasAdd?
deep_fm/dense_2/ReluRelu deep_fm/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
deep_fm/dense_2/Relul
deep_fm/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
deep_fm/concat/axis?
deep_fm/concatConcatV2 deep_fm/fm_layer/concat:output:0"deep_fm/dense_2/Relu:activations:0deep_fm/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
deep_fm/concat?
%deep_fm/dense_3/MatMul/ReadVariableOpReadVariableOp.deep_fm_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%deep_fm/dense_3/MatMul/ReadVariableOp?
deep_fm/dense_3/MatMulMatMuldeep_fm/concat:output:0-deep_fm/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
deep_fm/dense_3/MatMul?
&deep_fm/dense_3/BiasAdd/ReadVariableOpReadVariableOp/deep_fm_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&deep_fm/dense_3/BiasAdd/ReadVariableOp?
deep_fm/dense_3/BiasAddBiasAdd deep_fm/dense_3/MatMul:product:0.deep_fm/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
deep_fm/dense_3/BiasAdd?
deep_fm/dense_3/SigmoidSigmoid deep_fm/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
deep_fm/dense_3/Sigmoid?
deep_fm/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
deep_fm/Reshape_1/shape?
deep_fm/Reshape_1Reshapedeep_fm/dense_3/Sigmoid:y:0 deep_fm/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
deep_fm/Reshape_1q
IdentityIdentitydeep_fm/Reshape_1:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity?
NoOpNoOp%^deep_fm/dense/BiasAdd/ReadVariableOp$^deep_fm/dense/MatMul/ReadVariableOp'^deep_fm/dense_1/BiasAdd/ReadVariableOp&^deep_fm/dense_1/MatMul/ReadVariableOp'^deep_fm/dense_2/BiasAdd/ReadVariableOp&^deep_fm/dense_2/MatMul/ReadVariableOp'^deep_fm/dense_3/BiasAdd/ReadVariableOp&^deep_fm/dense_3/MatMul/ReadVariableOp&^deep_fm/fm_layer/Mul_1/ReadVariableOp"^deep_fm/fm_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2L
$deep_fm/dense/BiasAdd/ReadVariableOp$deep_fm/dense/BiasAdd/ReadVariableOp2J
#deep_fm/dense/MatMul/ReadVariableOp#deep_fm/dense/MatMul/ReadVariableOp2P
&deep_fm/dense_1/BiasAdd/ReadVariableOp&deep_fm/dense_1/BiasAdd/ReadVariableOp2N
%deep_fm/dense_1/MatMul/ReadVariableOp%deep_fm/dense_1/MatMul/ReadVariableOp2P
&deep_fm/dense_2/BiasAdd/ReadVariableOp&deep_fm/dense_2/BiasAdd/ReadVariableOp2N
%deep_fm/dense_2/MatMul/ReadVariableOp%deep_fm/dense_2/MatMul/ReadVariableOp2P
&deep_fm/dense_3/BiasAdd/ReadVariableOp&deep_fm/dense_3/BiasAdd/ReadVariableOp2N
%deep_fm/dense_3/MatMul/ReadVariableOp%deep_fm/dense_3/MatMul/ReadVariableOp2N
%deep_fm/fm_layer/Mul_1/ReadVariableOp%deep_fm/fm_layer/Mul_1/ReadVariableOp2F
!deep_fm/fm_layer/embedding_lookup!deep_fm/fm_layer/embedding_lookup:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?[
?
B__inference_deep_fm_layer_call_and_return_conditional_losses_40417
input_11
fm_layer_embedding_lookup_40354:5
&fm_layer_mul_1_readvariableop_resource:	?7
$dense_matmul_readvariableop_resource:	?@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?fm_layer/Mul_1/ReadVariableOp?fm_layer/embedding_lookup?
fm_layer/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????>     2
fm_layer/Reshape/shape?
fm_layer/ReshapeReshapeinput_1fm_layer/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
fm_layer/Reshape?
fm_layer/embedding_lookup/idsConst*
_output_shapes	
:?*
dtype0*?

value?
B?
?"?	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                2
fm_layer/embedding_lookup/ids?
fm_layer/embedding_lookupResourceGatherfm_layer_embedding_lookup_40354&fm_layer/embedding_lookup/ids:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*2
_class(
&$loc:@fm_layer/embedding_lookup/40354*
_output_shapes
:	?*
dtype02
fm_layer/embedding_lookup?
"fm_layer/embedding_lookup/IdentityIdentity"fm_layer/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@fm_layer/embedding_lookup/40354*
_output_shapes
:	?2$
"fm_layer/embedding_lookup/Identity?
$fm_layer/embedding_lookup/Identity_1Identity+fm_layer/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	?2&
$fm_layer/embedding_lookup/Identity_1?
fm_layer/MulMulfm_layer/Reshape:output:0-fm_layer/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
fm_layer/Mul?
fm_layer/Mul_1/ReadVariableOpReadVariableOp&fm_layer_mul_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
fm_layer/Mul_1/ReadVariableOp?
fm_layer/Mul_1Mul%fm_layer/Mul_1/ReadVariableOp:value:0input_1*
T0*(
_output_shapes
:??????????2
fm_layer/Mul_1?
fm_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
fm_layer/Sum/reduction_indices?
fm_layer/SumSumfm_layer/Mul_1:z:0'fm_layer/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Sum?
 fm_layer/Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2"
 fm_layer/Sum_1/reduction_indices?
fm_layer/Sum_1Sumfm_layer/Mul:z:0)fm_layer/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Sum_1s
fm_layer/SquareSquarefm_layer/Sum_1:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Squarey
fm_layer/Square_1Squarefm_layer/Mul:z:0*
T0*,
_output_shapes
:??????????2
fm_layer/Square_1?
 fm_layer/Sum_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2"
 fm_layer/Sum_2/reduction_indices?
fm_layer/Sum_2Sumfm_layer/Square_1:y:0)fm_layer/Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Sum_2
fm_layer/SubSubfm_layer/Square:y:0fm_layer/Sum_2:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Subi
fm_layer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
fm_layer/mul_2/x?
fm_layer/mul_2Mulfm_layer/mul_2/x:output:0fm_layer/Sub:z:0*
T0*#
_output_shapes
:?????????2
fm_layer/mul_2?
fm_layer/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
fm_layer/Reshape_1/shape?
fm_layer/Reshape_1Reshapefm_layer/Sum:output:0!fm_layer/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
fm_layer/Reshape_1?
fm_layer/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
fm_layer/Reshape_2/shape?
fm_layer/Reshape_2Reshapefm_layer/mul_2:z:0!fm_layer/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
fm_layer/Reshape_2n
fm_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
fm_layer/concat/axis?
fm_layer/concatConcatV2fm_layer/Reshape_1:output:0fm_layer/Reshape_2:output:0fm_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
fm_layer/concato
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Reshape/shapez
ReshapeReshapefm_layer/Mul:z:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relu|
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Relu?
dropout_1/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
dropout_1/Identity?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Relu\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2fm_layer/concat:output:0dense_2/Relu:activations:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulconcat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shape~
	Reshape_1Reshapedense_3/Sigmoid:y:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1i
IdentityIdentityReshape_1:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^fm_layer/Mul_1/ReadVariableOp^fm_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2>
fm_layer/Mul_1/ReadVariableOpfm_layer/Mul_1/ReadVariableOp26
fm_layer/embedding_lookupfm_layer/embedding_lookup:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
`
'__inference_dropout_layer_call_fn_40696

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_399472
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
'__inference_deep_fm_layer_call_fn_40550

inputs
unknown:
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_deep_fm_layer_call_and_return_conditional_losses_398512
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_39810

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_dense_3_layer_call_and_return_conditional_losses_39842

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_40669

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_397752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
C__inference_fm_layer_layer_call_and_return_conditional_losses_39755

inputs(
embedding_lookup_39727:,
mul_1_readvariableop_resource:	?
identity

identity_1??Mul_1/ReadVariableOp?embedding_lookups
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????>     2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshape?

embedding_lookup/idsConst*
_output_shapes	
:?*
dtype0*?

value?
B?
?"?	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                2
embedding_lookup/ids?
embedding_lookupResourceGatherembedding_lookup_39727embedding_lookup/ids:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/39727*
_output_shapes
:	?*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/39727*
_output_shapes
:	?2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	?2
embedding_lookup/Identity_1?
MulMulReshape:output:0$embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
Mul?
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Mul_1/ReadVariableOpn
Mul_1MulMul_1/ReadVariableOp:value:0inputs*
T0*(
_output_shapes
:??????????2
Mul_1p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesj
SumSum	Mul_1:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum?
Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_1/reduction_indicesn
Sum_1SumMul:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_1X
SquareSquareSum_1:output:0*
T0*#
_output_shapes
:?????????2
Square^
Square_1SquareMul:z:0*
T0*,
_output_shapes
:??????????2

Square_1?
Sum_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_2/reduction_indicess
Sum_2SumSquare_1:y:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_2[
SubSub
Square:y:0Sum_2:output:0*
T0*#
_output_shapes
:?????????2
SubW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_2/x^
mul_2Mulmul_2/x:output:0Sub:z:0*
T0*#
_output_shapes
:?????????2
mul_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_1/shape{
	Reshape_1ReshapeSum:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_2/shapex
	Reshape_2Reshape	mul_2:z:0Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_2\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1IdentityMul:z:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity_1x
NoOpNoOp^Mul_1/ReadVariableOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
'__inference_deep_fm_layer_call_fn_40575

inputs
unknown:
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_deep_fm_layer_call_and_return_conditional_losses_400372
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_dense_3_layer_call_fn_40783

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_398422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_40674

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?m
?
B__inference_deep_fm_layer_call_and_return_conditional_losses_40348

inputs1
fm_layer_embedding_lookup_40271:5
&fm_layer_mul_1_readvariableop_resource:	?7
$dense_matmul_readvariableop_resource:	?@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?fm_layer/Mul_1/ReadVariableOp?fm_layer/embedding_lookup?
fm_layer/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????>     2
fm_layer/Reshape/shape?
fm_layer/ReshapeReshapeinputsfm_layer/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
fm_layer/Reshape?
fm_layer/embedding_lookup/idsConst*
_output_shapes	
:?*
dtype0*?

value?
B?
?"?	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                2
fm_layer/embedding_lookup/ids?
fm_layer/embedding_lookupResourceGatherfm_layer_embedding_lookup_40271&fm_layer/embedding_lookup/ids:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*2
_class(
&$loc:@fm_layer/embedding_lookup/40271*
_output_shapes
:	?*
dtype02
fm_layer/embedding_lookup?
"fm_layer/embedding_lookup/IdentityIdentity"fm_layer/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@fm_layer/embedding_lookup/40271*
_output_shapes
:	?2$
"fm_layer/embedding_lookup/Identity?
$fm_layer/embedding_lookup/Identity_1Identity+fm_layer/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	?2&
$fm_layer/embedding_lookup/Identity_1?
fm_layer/MulMulfm_layer/Reshape:output:0-fm_layer/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
fm_layer/Mul?
fm_layer/Mul_1/ReadVariableOpReadVariableOp&fm_layer_mul_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
fm_layer/Mul_1/ReadVariableOp?
fm_layer/Mul_1Mul%fm_layer/Mul_1/ReadVariableOp:value:0inputs*
T0*(
_output_shapes
:??????????2
fm_layer/Mul_1?
fm_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
fm_layer/Sum/reduction_indices?
fm_layer/SumSumfm_layer/Mul_1:z:0'fm_layer/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Sum?
 fm_layer/Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2"
 fm_layer/Sum_1/reduction_indices?
fm_layer/Sum_1Sumfm_layer/Mul:z:0)fm_layer/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Sum_1s
fm_layer/SquareSquarefm_layer/Sum_1:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Squarey
fm_layer/Square_1Squarefm_layer/Mul:z:0*
T0*,
_output_shapes
:??????????2
fm_layer/Square_1?
 fm_layer/Sum_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2"
 fm_layer/Sum_2/reduction_indices?
fm_layer/Sum_2Sumfm_layer/Square_1:y:0)fm_layer/Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Sum_2
fm_layer/SubSubfm_layer/Square:y:0fm_layer/Sum_2:output:0*
T0*#
_output_shapes
:?????????2
fm_layer/Subi
fm_layer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
fm_layer/mul_2/x?
fm_layer/mul_2Mulfm_layer/mul_2/x:output:0fm_layer/Sub:z:0*
T0*#
_output_shapes
:?????????2
fm_layer/mul_2?
fm_layer/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
fm_layer/Reshape_1/shape?
fm_layer/Reshape_1Reshapefm_layer/Sum:output:0!fm_layer/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
fm_layer/Reshape_1?
fm_layer/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
fm_layer/Reshape_2/shape?
fm_layer/Reshape_2Reshapefm_layer/mul_2:z:0!fm_layer/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
fm_layer/Reshape_2n
fm_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
fm_layer/concat/axis?
fm_layer/concatConcatV2fm_layer/Reshape_1:output:0fm_layer/Reshape_2:output:0fm_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
fm_layer/concato
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Reshape/shapez
ReshapeReshapefm_layer/Mul:z:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMuldense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_1/dropout/Mul|
dropout_1/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_1/dropout/Mul_1?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Relu\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2fm_layer/concat:output:0dense_2/Relu:activations:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concat?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulconcat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shape~
	Reshape_1Reshapedense_3/Sigmoid:y:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1i
IdentityIdentityReshape_1:output:0^NoOp*
T0*#
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^fm_layer/Mul_1/ReadVariableOp^fm_layer/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2>
fm_layer/Mul_1/ReadVariableOpfm_layer/Mul_1/ReadVariableOp26
fm_layer/embedding_lookupfm_layer/embedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_dense_layer_call_and_return_conditional_losses_39775

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????8
output_1,
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?|
?
field_index
fm_layer
layers1
dropout1
layers2
dropout2
layers3
	final
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
*]&call_and_return_all_conditional_losses
^_default_save_signature
___call__"
_tf_keras_model
 "
trackable_list_wrapper
?
field_index
w
V
trainable_variables
	variables
regularization_losses
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"
_tf_keras_layer
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*b&call_and_return_all_conditional_losses
c__call__"
_tf_keras_layer
?
trainable_variables
	variables
regularization_losses
	keras_api
*d&call_and_return_all_conditional_losses
e__call__"
_tf_keras_layer
?

kernel
 bias
!trainable_variables
"	variables
#regularization_losses
$	keras_api
*f&call_and_return_all_conditional_losses
g__call__"
_tf_keras_layer
?
%trainable_variables
&	variables
'regularization_losses
(	keras_api
*h&call_and_return_all_conditional_losses
i__call__"
_tf_keras_layer
?

)kernel
*bias
+trainable_variables
,	variables
-regularization_losses
.	keras_api
*j&call_and_return_all_conditional_losses
k__call__"
_tf_keras_layer
?

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
*l&call_and_return_all_conditional_losses
m__call__"
_tf_keras_layer
f
0
1
2
3
4
 5
)6
*7
/8
09"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
 5
)6
*7
/8
09"
trackable_list_wrapper
?
	trainable_variables
5layer_regularization_losses

regularization_losses
6layer_metrics
7non_trainable_variables
8metrics
	variables

9layers
___call__
^_default_save_signature
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
,
nserving_default"
signature_map
 "
trackable_list_wrapper
:?2w
:2V
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
:layer_regularization_losses
regularization_losses
;layer_metrics
<non_trainable_variables
=metrics

>layers
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
':%	?@2deep_fm/dense/kernel
 :@2deep_fm/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
?layer_regularization_losses
regularization_losses
@layer_metrics
Anon_trainable_variables
Bmetrics

Clayers
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
Dlayer_regularization_losses
regularization_losses
Elayer_metrics
Fnon_trainable_variables
Gmetrics

Hlayers
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
(:&@2deep_fm/dense_1/kernel
": 2deep_fm/dense_1/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
!trainable_variables
"	variables
Ilayer_regularization_losses
#regularization_losses
Jlayer_metrics
Knon_trainable_variables
Lmetrics

Mlayers
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
%trainable_variables
&	variables
Nlayer_regularization_losses
'regularization_losses
Olayer_metrics
Pnon_trainable_variables
Qmetrics

Rlayers
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
(:&2deep_fm/dense_2/kernel
": 2deep_fm/dense_2/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
+trainable_variables
,	variables
Slayer_regularization_losses
-regularization_losses
Tlayer_metrics
Unon_trainable_variables
Vmetrics

Wlayers
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
(:&2deep_fm/dense_3/kernel
": 2deep_fm/dense_3/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
1trainable_variables
2	variables
Xlayer_regularization_losses
3regularization_losses
Ylayer_metrics
Znon_trainable_variables
[metrics

\layers
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
B__inference_deep_fm_layer_call_and_return_conditional_losses_40265
B__inference_deep_fm_layer_call_and_return_conditional_losses_40348
B__inference_deep_fm_layer_call_and_return_conditional_losses_40417
B__inference_deep_fm_layer_call_and_return_conditional_losses_40500?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
 __inference__wrapped_model_39714?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_1??????????
?2?
'__inference_deep_fm_layer_call_fn_40525
'__inference_deep_fm_layer_call_fn_40550
'__inference_deep_fm_layer_call_fn_40575
'__inference_deep_fm_layer_call_fn_40600?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_fm_layer_layer_call_and_return_conditional_losses_40638?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_fm_layer_layer_call_fn_40649?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_40660?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_40669?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_40674
B__inference_dropout_layer_call_and_return_conditional_losses_40686?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_40691
'__inference_dropout_layer_call_fn_40696?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_40707?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_40716?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_40721
D__inference_dropout_1_layer_call_and_return_conditional_losses_40733?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_1_layer_call_fn_40738
)__inference_dropout_1_layer_call_fn_40743?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_40754?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_40763?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_40774?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_3_layer_call_fn_40783?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_40196input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_39714p
 )*/01?.
'?$
"?
input_1??????????
? "/?,
*
output_1?
output_1??????????
B__inference_deep_fm_layer_call_and_return_conditional_losses_40265e
 )*/04?1
*?'
!?
inputs??????????
p 
? "!?
?
0?????????
? ?
B__inference_deep_fm_layer_call_and_return_conditional_losses_40348e
 )*/04?1
*?'
!?
inputs??????????
p
? "!?
?
0?????????
? ?
B__inference_deep_fm_layer_call_and_return_conditional_losses_40417f
 )*/05?2
+?(
"?
input_1??????????
p 
? "!?
?
0?????????
? ?
B__inference_deep_fm_layer_call_and_return_conditional_losses_40500f
 )*/05?2
+?(
"?
input_1??????????
p
? "!?
?
0?????????
? ?
'__inference_deep_fm_layer_call_fn_40525Y
 )*/05?2
+?(
"?
input_1??????????
p 
? "???????????
'__inference_deep_fm_layer_call_fn_40550X
 )*/04?1
*?'
!?
inputs??????????
p 
? "???????????
'__inference_deep_fm_layer_call_fn_40575X
 )*/04?1
*?'
!?
inputs??????????
p
? "???????????
'__inference_deep_fm_layer_call_fn_40600Y
 )*/05?2
+?(
"?
input_1??????????
p
? "???????????
B__inference_dense_1_layer_call_and_return_conditional_losses_40707\ /?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_40716O /?,
%?"
 ?
inputs?????????@
? "???????????
B__inference_dense_2_layer_call_and_return_conditional_losses_40754\)*/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_2_layer_call_fn_40763O)*/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_3_layer_call_and_return_conditional_losses_40774\/0/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_3_layer_call_fn_40783O/0/?,
%?"
 ?
inputs?????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_40660]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? y
%__inference_dense_layer_call_fn_40669P0?-
&?#
!?
inputs??????????
? "??????????@?
D__inference_dropout_1_layer_call_and_return_conditional_losses_40721\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_40733\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? |
)__inference_dropout_1_layer_call_fn_40738O3?0
)?&
 ?
inputs?????????
p 
? "??????????|
)__inference_dropout_1_layer_call_fn_40743O3?0
)?&
 ?
inputs?????????
p
? "???????????
B__inference_dropout_layer_call_and_return_conditional_losses_40674\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_40686\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? z
'__inference_dropout_layer_call_fn_40691O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@z
'__inference_dropout_layer_call_fn_40696O3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
C__inference_fm_layer_layer_call_and_return_conditional_losses_40638?0?-
&?#
!?
inputs??????????
? "P?M
F?C
?
0/0?????????
"?
0/1??????????
? ?
(__inference_fm_layer_layer_call_fn_40649z0?-
&?#
!?
inputs??????????
? "B??
?
0?????????
 ?
1???????????
#__inference_signature_wrapper_40196{
 )*/0<?9
? 
2?/
-
input_1"?
input_1??????????"/?,
*
output_1?
output_1?????????