юб
Ц ь
:
Add
x"T
y"T
z"T"
Ttype:
2	
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
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
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z
Р
!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	Р
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
;
Minimum
x"T
y"T
z"T"
Ttype:

2	Р
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
2
NextIteration	
data"T
output"T"	
Ttype
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
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:И
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetypeИ
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
TtypeИ
9
TensorArraySizeV3

handle
flow_in
sizeИ
▐
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring И
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
TtypeИ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И"serve*1.8.02b'v1.8.0-8-g23c218785e'р╚

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
П
global_step
VariableV2*
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container 
▓
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step
r
inputPlaceholder*
dtype0*,
_output_shapes
:         ш*!
shape:         ш
\
Conv1d_LSTM_model/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
c
!Conv1d_LSTM_model/rnn/range/startConst*
dtype0*
_output_shapes
: *
value	B :
c
!Conv1d_LSTM_model/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
о
Conv1d_LSTM_model/rnn/rangeRange!Conv1d_LSTM_model/rnn/range/startConv1d_LSTM_model/rnn/Rank!Conv1d_LSTM_model/rnn/range/delta*

Tidx0*
_output_shapes
:
v
%Conv1d_LSTM_model/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
c
!Conv1d_LSTM_model/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╔
Conv1d_LSTM_model/rnn/concatConcatV2%Conv1d_LSTM_model/rnn/concat/values_0Conv1d_LSTM_model/rnn/range!Conv1d_LSTM_model/rnn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Х
Conv1d_LSTM_model/rnn/transpose	TransposeinputConv1d_LSTM_model/rnn/concat*
T0*,
_output_shapes
:ш         *
Tperm0
z
Conv1d_LSTM_model/rnn/ShapeShapeConv1d_LSTM_model/rnn/transpose*
_output_shapes
:*
T0*
out_type0
s
)Conv1d_LSTM_model/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
u
+Conv1d_LSTM_model/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+Conv1d_LSTM_model/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ч
#Conv1d_LSTM_model/rnn/strided_sliceStridedSliceConv1d_LSTM_model/rnn/Shape)Conv1d_LSTM_model/rnn/strided_slice/stack+Conv1d_LSTM_model/rnn/strided_slice/stack_1+Conv1d_LSTM_model/rnn/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
x
6Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
╬
2Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims
ExpandDims#Conv1d_LSTM_model/rnn/strided_slice6Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
x
-Conv1d_LSTM_model/rnn/LSTMCellZeroState/ConstConst*
valueB:╚*
dtype0*
_output_shapes
:
u
3Conv1d_LSTM_model/rnn/LSTMCellZeroState/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
М
.Conv1d_LSTM_model/rnn/LSTMCellZeroState/concatConcatV22Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims-Conv1d_LSTM_model/rnn/LSTMCellZeroState/Const3Conv1d_LSTM_model/rnn/LSTMCellZeroState/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
x
3Conv1d_LSTM_model/rnn/LSTMCellZeroState/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
▀
-Conv1d_LSTM_model/rnn/LSTMCellZeroState/zerosFill.Conv1d_LSTM_model/rnn/LSTMCellZeroState/concat3Conv1d_LSTM_model/rnn/LSTMCellZeroState/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ╚
z
8Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
╥
4Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims_1
ExpandDims#Conv1d_LSTM_model/rnn/strided_slice8Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims_1/dim*
_output_shapes
:*

Tdim0*
T0
z
/Conv1d_LSTM_model/rnn/LSTMCellZeroState/Const_1Const*
valueB:╚*
dtype0*
_output_shapes
:
z
8Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
╥
4Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims_2
ExpandDims#Conv1d_LSTM_model/rnn/strided_slice8Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims_2/dim*
_output_shapes
:*

Tdim0*
T0
z
/Conv1d_LSTM_model/rnn/LSTMCellZeroState/Const_2Const*
valueB:╚*
dtype0*
_output_shapes
:
w
5Conv1d_LSTM_model/rnn/LSTMCellZeroState/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ф
0Conv1d_LSTM_model/rnn/LSTMCellZeroState/concat_1ConcatV24Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims_2/Conv1d_LSTM_model/rnn/LSTMCellZeroState/Const_25Conv1d_LSTM_model/rnn/LSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
z
5Conv1d_LSTM_model/rnn/LSTMCellZeroState/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
х
/Conv1d_LSTM_model/rnn/LSTMCellZeroState/zeros_1Fill0Conv1d_LSTM_model/rnn/LSTMCellZeroState/concat_15Conv1d_LSTM_model/rnn/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0*(
_output_shapes
:         ╚
z
8Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
╥
4Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims_3
ExpandDims#Conv1d_LSTM_model/rnn/strided_slice8Conv1d_LSTM_model/rnn/LSTMCellZeroState/ExpandDims_3/dim*

Tdim0*
T0*
_output_shapes
:
z
/Conv1d_LSTM_model/rnn/LSTMCellZeroState/Const_3Const*
valueB:╚*
dtype0*
_output_shapes
:
|
Conv1d_LSTM_model/rnn/Shape_1ShapeConv1d_LSTM_model/rnn/transpose*
out_type0*
_output_shapes
:*
T0
u
+Conv1d_LSTM_model/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
-Conv1d_LSTM_model/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-Conv1d_LSTM_model/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ё
%Conv1d_LSTM_model/rnn/strided_slice_1StridedSliceConv1d_LSTM_model/rnn/Shape_1+Conv1d_LSTM_model/rnn/strided_slice_1/stack-Conv1d_LSTM_model/rnn/strided_slice_1/stack_1-Conv1d_LSTM_model/rnn/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
|
Conv1d_LSTM_model/rnn/Shape_2ShapeConv1d_LSTM_model/rnn/transpose*
T0*
out_type0*
_output_shapes
:
u
+Conv1d_LSTM_model/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
w
-Conv1d_LSTM_model/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-Conv1d_LSTM_model/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ё
%Conv1d_LSTM_model/rnn/strided_slice_2StridedSliceConv1d_LSTM_model/rnn/Shape_2+Conv1d_LSTM_model/rnn/strided_slice_2/stack-Conv1d_LSTM_model/rnn/strided_slice_2/stack_1-Conv1d_LSTM_model/rnn/strided_slice_2/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask 
f
$Conv1d_LSTM_model/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
м
 Conv1d_LSTM_model/rnn/ExpandDims
ExpandDims%Conv1d_LSTM_model/rnn/strided_slice_2$Conv1d_LSTM_model/rnn/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
f
Conv1d_LSTM_model/rnn/ConstConst*
valueB:╚*
dtype0*
_output_shapes
:
e
#Conv1d_LSTM_model/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╚
Conv1d_LSTM_model/rnn/concat_1ConcatV2 Conv1d_LSTM_model/rnn/ExpandDimsConv1d_LSTM_model/rnn/Const#Conv1d_LSTM_model/rnn/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
f
!Conv1d_LSTM_model/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
л
Conv1d_LSTM_model/rnn/zerosFillConv1d_LSTM_model/rnn/concat_1!Conv1d_LSTM_model/rnn/zeros/Const*

index_type0*(
_output_shapes
:         ╚*
T0
\
Conv1d_LSTM_model/rnn/timeConst*
_output_shapes
: *
value	B : *
dtype0
║
!Conv1d_LSTM_model/rnn/TensorArrayTensorArrayV3%Conv1d_LSTM_model/rnn/strided_slice_1*
dtype0*
_output_shapes

:: *%
element_shape:         ╚*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*A
tensor_array_name,*Conv1d_LSTM_model/rnn/dynamic_rnn/output_0
║
#Conv1d_LSTM_model/rnn/TensorArray_1TensorArrayV3%Conv1d_LSTM_model/rnn/strided_slice_1*@
tensor_array_name+)Conv1d_LSTM_model/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:         *
dynamic_size( *
clear_after_read(*
identical_element_shapes(
Н
.Conv1d_LSTM_model/rnn/TensorArrayUnstack/ShapeShapeConv1d_LSTM_model/rnn/transpose*
T0*
out_type0*
_output_shapes
:
Ж
<Conv1d_LSTM_model/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
И
>Conv1d_LSTM_model/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
И
>Conv1d_LSTM_model/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╞
6Conv1d_LSTM_model/rnn/TensorArrayUnstack/strided_sliceStridedSlice.Conv1d_LSTM_model/rnn/TensorArrayUnstack/Shape<Conv1d_LSTM_model/rnn/TensorArrayUnstack/strided_slice/stack>Conv1d_LSTM_model/rnn/TensorArrayUnstack/strided_slice/stack_1>Conv1d_LSTM_model/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
v
4Conv1d_LSTM_model/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
v
4Conv1d_LSTM_model/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
М
.Conv1d_LSTM_model/rnn/TensorArrayUnstack/rangeRange4Conv1d_LSTM_model/rnn/TensorArrayUnstack/range/start6Conv1d_LSTM_model/rnn/TensorArrayUnstack/strided_slice4Conv1d_LSTM_model/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:         *

Tidx0
┌
PConv1d_LSTM_model/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3#Conv1d_LSTM_model/rnn/TensorArray_1.Conv1d_LSTM_model/rnn/TensorArrayUnstack/rangeConv1d_LSTM_model/rnn/transpose%Conv1d_LSTM_model/rnn/TensorArray_1:1*2
_class(
&$loc:@Conv1d_LSTM_model/rnn/transpose*
_output_shapes
: *
T0
a
Conv1d_LSTM_model/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
С
Conv1d_LSTM_model/rnn/MaximumMaximumConv1d_LSTM_model/rnn/Maximum/x%Conv1d_LSTM_model/rnn/strided_slice_1*
T0*
_output_shapes
: 
П
Conv1d_LSTM_model/rnn/MinimumMinimum%Conv1d_LSTM_model/rnn/strided_slice_1Conv1d_LSTM_model/rnn/Maximum*
T0*
_output_shapes
: 
o
-Conv1d_LSTM_model/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
у
!Conv1d_LSTM_model/rnn/while/EnterEnter-Conv1d_LSTM_model/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *9

frame_name+)Conv1d_LSTM_model/rnn/while/while_context
╥
#Conv1d_LSTM_model/rnn/while/Enter_1EnterConv1d_LSTM_model/rnn/time*
parallel_iterations *
_output_shapes
: *9

frame_name+)Conv1d_LSTM_model/rnn/while/while_context*
T0*
is_constant( 
█
#Conv1d_LSTM_model/rnn/while/Enter_2Enter#Conv1d_LSTM_model/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *9

frame_name+)Conv1d_LSTM_model/rnn/while/while_context
ў
#Conv1d_LSTM_model/rnn/while/Enter_3Enter-Conv1d_LSTM_model/rnn/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         ╚*9

frame_name+)Conv1d_LSTM_model/rnn/while/while_context
∙
#Conv1d_LSTM_model/rnn/while/Enter_4Enter/Conv1d_LSTM_model/rnn/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         ╚*9

frame_name+)Conv1d_LSTM_model/rnn/while/while_context
д
!Conv1d_LSTM_model/rnn/while/MergeMerge!Conv1d_LSTM_model/rnn/while/Enter)Conv1d_LSTM_model/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
к
#Conv1d_LSTM_model/rnn/while/Merge_1Merge#Conv1d_LSTM_model/rnn/while/Enter_1+Conv1d_LSTM_model/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
к
#Conv1d_LSTM_model/rnn/while/Merge_2Merge#Conv1d_LSTM_model/rnn/while/Enter_2+Conv1d_LSTM_model/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
╝
#Conv1d_LSTM_model/rnn/while/Merge_3Merge#Conv1d_LSTM_model/rnn/while/Enter_3+Conv1d_LSTM_model/rnn/while/NextIteration_3*
N**
_output_shapes
:         ╚: *
T0
╝
#Conv1d_LSTM_model/rnn/while/Merge_4Merge#Conv1d_LSTM_model/rnn/while/Enter_4+Conv1d_LSTM_model/rnn/while/NextIteration_4*
T0*
N**
_output_shapes
:         ╚: 
Ф
 Conv1d_LSTM_model/rnn/while/LessLess!Conv1d_LSTM_model/rnn/while/Merge&Conv1d_LSTM_model/rnn/while/Less/Enter*
T0*
_output_shapes
: 
р
&Conv1d_LSTM_model/rnn/while/Less/EnterEnter%Conv1d_LSTM_model/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *9

frame_name+)Conv1d_LSTM_model/rnn/while/while_context
Ъ
"Conv1d_LSTM_model/rnn/while/Less_1Less#Conv1d_LSTM_model/rnn/while/Merge_1(Conv1d_LSTM_model/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
┌
(Conv1d_LSTM_model/rnn/while/Less_1/EnterEnterConv1d_LSTM_model/rnn/Minimum*
parallel_iterations *
_output_shapes
: *9

frame_name+)Conv1d_LSTM_model/rnn/while/while_context*
T0*
is_constant(
Т
&Conv1d_LSTM_model/rnn/while/LogicalAnd
LogicalAnd Conv1d_LSTM_model/rnn/while/Less"Conv1d_LSTM_model/rnn/while/Less_1*
_output_shapes
: 
p
$Conv1d_LSTM_model/rnn/while/LoopCondLoopCond&Conv1d_LSTM_model/rnn/while/LogicalAnd*
_output_shapes
: 
╬
"Conv1d_LSTM_model/rnn/while/SwitchSwitch!Conv1d_LSTM_model/rnn/while/Merge$Conv1d_LSTM_model/rnn/while/LoopCond*
T0*4
_class*
(&loc:@Conv1d_LSTM_model/rnn/while/Merge*
_output_shapes
: : 
╘
$Conv1d_LSTM_model/rnn/while/Switch_1Switch#Conv1d_LSTM_model/rnn/while/Merge_1$Conv1d_LSTM_model/rnn/while/LoopCond*
T0*6
_class,
*(loc:@Conv1d_LSTM_model/rnn/while/Merge_1*
_output_shapes
: : 
╘
$Conv1d_LSTM_model/rnn/while/Switch_2Switch#Conv1d_LSTM_model/rnn/while/Merge_2$Conv1d_LSTM_model/rnn/while/LoopCond*
T0*6
_class,
*(loc:@Conv1d_LSTM_model/rnn/while/Merge_2*
_output_shapes
: : 
°
$Conv1d_LSTM_model/rnn/while/Switch_3Switch#Conv1d_LSTM_model/rnn/while/Merge_3$Conv1d_LSTM_model/rnn/while/LoopCond*
T0*6
_class,
*(loc:@Conv1d_LSTM_model/rnn/while/Merge_3*<
_output_shapes*
(:         ╚:         ╚
°
$Conv1d_LSTM_model/rnn/while/Switch_4Switch#Conv1d_LSTM_model/rnn/while/Merge_4$Conv1d_LSTM_model/rnn/while/LoopCond*
T0*6
_class,
*(loc:@Conv1d_LSTM_model/rnn/while/Merge_4*<
_output_shapes*
(:         ╚:         ╚
w
$Conv1d_LSTM_model/rnn/while/IdentityIdentity$Conv1d_LSTM_model/rnn/while/Switch:1*
_output_shapes
: *
T0
{
&Conv1d_LSTM_model/rnn/while/Identity_1Identity&Conv1d_LSTM_model/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
{
&Conv1d_LSTM_model/rnn/while/Identity_2Identity&Conv1d_LSTM_model/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
Н
&Conv1d_LSTM_model/rnn/while/Identity_3Identity&Conv1d_LSTM_model/rnn/while/Switch_3:1*
T0*(
_output_shapes
:         ╚
Н
&Conv1d_LSTM_model/rnn/while/Identity_4Identity&Conv1d_LSTM_model/rnn/while/Switch_4:1*(
_output_shapes
:         ╚*
T0
К
!Conv1d_LSTM_model/rnn/while/add/yConst%^Conv1d_LSTM_model/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Р
Conv1d_LSTM_model/rnn/while/addAdd$Conv1d_LSTM_model/rnn/while/Identity!Conv1d_LSTM_model/rnn/while/add/y*
T0*
_output_shapes
: 
М
-Conv1d_LSTM_model/rnn/while/TensorArrayReadV3TensorArrayReadV33Conv1d_LSTM_model/rnn/while/TensorArrayReadV3/Enter&Conv1d_LSTM_model/rnn/while/Identity_15Conv1d_LSTM_model/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:         
я
3Conv1d_LSTM_model/rnn/while/TensorArrayReadV3/EnterEnter#Conv1d_LSTM_model/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)Conv1d_LSTM_model/rnn/while/while_context
Ъ
5Conv1d_LSTM_model/rnn/while/TensorArrayReadV3/Enter_1EnterPConv1d_LSTM_model/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *9

frame_name+)Conv1d_LSTM_model/rnn/while/while_context
╙
GConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*9
_class/
-+loc:@Conv1d_LSTM_model/rnn/lstm_cell/kernel*
valueB"╩      
┼
EConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@Conv1d_LSTM_model/rnn/lstm_cell/kernel*
valueB
 *аzЮ╜
┼
EConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/maxConst*9
_class/
-+loc:@Conv1d_LSTM_model/rnn/lstm_cell/kernel*
valueB
 *аzЮ=*
dtype0*
_output_shapes
: 
╡
OConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformGConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/shape*

seed *
T0*9
_class/
-+loc:@Conv1d_LSTM_model/rnn/lstm_cell/kernel*
seed2 *
dtype0* 
_output_shapes
:
╩а
╢
EConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/subSubEConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/maxEConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@Conv1d_LSTM_model/rnn/lstm_cell/kernel*
_output_shapes
: 
╩
EConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/mulMulOConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformEConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*9
_class/
-+loc:@Conv1d_LSTM_model/rnn/lstm_cell/kernel* 
_output_shapes
:
╩а
╝
AConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniformAddEConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/mulEConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@Conv1d_LSTM_model/rnn/lstm_cell/kernel* 
_output_shapes
:
╩а
┘
&Conv1d_LSTM_model/rnn/lstm_cell/kernel
VariableV2*
shared_name *9
_class/
-+loc:@Conv1d_LSTM_model/rnn/lstm_cell/kernel*
	container *
shape:
╩а*
dtype0* 
_output_shapes
:
╩а
▒
-Conv1d_LSTM_model/rnn/lstm_cell/kernel/AssignAssign&Conv1d_LSTM_model/rnn/lstm_cell/kernelAConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*9
_class/
-+loc:@Conv1d_LSTM_model/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
╩а
К
+Conv1d_LSTM_model/rnn/lstm_cell/kernel/readIdentity&Conv1d_LSTM_model/rnn/lstm_cell/kernel*
T0* 
_output_shapes
:
╩а
╛
6Conv1d_LSTM_model/rnn/lstm_cell/bias/Initializer/zerosConst*7
_class-
+)loc:@Conv1d_LSTM_model/rnn/lstm_cell/bias*
valueBа*    *
dtype0*
_output_shapes	
:а
╦
$Conv1d_LSTM_model/rnn/lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:а*
shared_name *7
_class-
+)loc:@Conv1d_LSTM_model/rnn/lstm_cell/bias*
	container *
shape:а
Ы
+Conv1d_LSTM_model/rnn/lstm_cell/bias/AssignAssign$Conv1d_LSTM_model/rnn/lstm_cell/bias6Conv1d_LSTM_model/rnn/lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@Conv1d_LSTM_model/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:а
Б
)Conv1d_LSTM_model/rnn/lstm_cell/bias/readIdentity$Conv1d_LSTM_model/rnn/lstm_cell/bias*
T0*
_output_shapes	
:а
Ъ
1Conv1d_LSTM_model/rnn/while/lstm_cell/concat/axisConst%^Conv1d_LSTM_model/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
К
,Conv1d_LSTM_model/rnn/while/lstm_cell/concatConcatV2-Conv1d_LSTM_model/rnn/while/TensorArrayReadV3&Conv1d_LSTM_model/rnn/while/Identity_41Conv1d_LSTM_model/rnn/while/lstm_cell/concat/axis*(
_output_shapes
:         ╩*

Tidx0*
T0*
N
ё
,Conv1d_LSTM_model/rnn/while/lstm_cell/MatMulMatMul,Conv1d_LSTM_model/rnn/while/lstm_cell/concat2Conv1d_LSTM_model/rnn/while/lstm_cell/MatMul/Enter*
T0*(
_output_shapes
:         а*
transpose_a( *
transpose_b( 
№
2Conv1d_LSTM_model/rnn/while/lstm_cell/MatMul/EnterEnter+Conv1d_LSTM_model/rnn/lstm_cell/kernel/read*
is_constant(*
parallel_iterations * 
_output_shapes
:
╩а*9

frame_name+)Conv1d_LSTM_model/rnn/while/while_context*
T0
х
-Conv1d_LSTM_model/rnn/while/lstm_cell/BiasAddBiasAdd,Conv1d_LSTM_model/rnn/while/lstm_cell/MatMul3Conv1d_LSTM_model/rnn/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:         а
Ў
3Conv1d_LSTM_model/rnn/while/lstm_cell/BiasAdd/EnterEnter)Conv1d_LSTM_model/rnn/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:а*9

frame_name+)Conv1d_LSTM_model/rnn/while/while_context
Ф
+Conv1d_LSTM_model/rnn/while/lstm_cell/ConstConst%^Conv1d_LSTM_model/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ю
5Conv1d_LSTM_model/rnn/while/lstm_cell/split/split_dimConst%^Conv1d_LSTM_model/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ъ
+Conv1d_LSTM_model/rnn/while/lstm_cell/splitSplit5Conv1d_LSTM_model/rnn/while/lstm_cell/split/split_dim-Conv1d_LSTM_model/rnn/while/lstm_cell/BiasAdd*d
_output_shapesR
P:         ╚:         ╚:         ╚:         ╚*
	num_split*
T0
Ч
+Conv1d_LSTM_model/rnn/while/lstm_cell/add/yConst%^Conv1d_LSTM_model/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  А?
┐
)Conv1d_LSTM_model/rnn/while/lstm_cell/addAdd-Conv1d_LSTM_model/rnn/while/lstm_cell/split:2+Conv1d_LSTM_model/rnn/while/lstm_cell/add/y*
T0*(
_output_shapes
:         ╚
Ц
-Conv1d_LSTM_model/rnn/while/lstm_cell/SigmoidSigmoid)Conv1d_LSTM_model/rnn/while/lstm_cell/add*
T0*(
_output_shapes
:         ╚
║
)Conv1d_LSTM_model/rnn/while/lstm_cell/mulMul-Conv1d_LSTM_model/rnn/while/lstm_cell/Sigmoid&Conv1d_LSTM_model/rnn/while/Identity_3*
T0*(
_output_shapes
:         ╚
Ъ
/Conv1d_LSTM_model/rnn/while/lstm_cell/Sigmoid_1Sigmoid+Conv1d_LSTM_model/rnn/while/lstm_cell/split*(
_output_shapes
:         ╚*
T0
Ф
*Conv1d_LSTM_model/rnn/while/lstm_cell/TanhTanh-Conv1d_LSTM_model/rnn/while/lstm_cell/split:1*(
_output_shapes
:         ╚*
T0
┬
+Conv1d_LSTM_model/rnn/while/lstm_cell/mul_1Mul/Conv1d_LSTM_model/rnn/while/lstm_cell/Sigmoid_1*Conv1d_LSTM_model/rnn/while/lstm_cell/Tanh*
T0*(
_output_shapes
:         ╚
╜
+Conv1d_LSTM_model/rnn/while/lstm_cell/add_1Add)Conv1d_LSTM_model/rnn/while/lstm_cell/mul+Conv1d_LSTM_model/rnn/while/lstm_cell/mul_1*(
_output_shapes
:         ╚*
T0
Ь
/Conv1d_LSTM_model/rnn/while/lstm_cell/Sigmoid_2Sigmoid-Conv1d_LSTM_model/rnn/while/lstm_cell/split:3*
T0*(
_output_shapes
:         ╚
Ф
,Conv1d_LSTM_model/rnn/while/lstm_cell/Tanh_1Tanh+Conv1d_LSTM_model/rnn/while/lstm_cell/add_1*
T0*(
_output_shapes
:         ╚
─
+Conv1d_LSTM_model/rnn/while/lstm_cell/mul_2Mul/Conv1d_LSTM_model/rnn/while/lstm_cell/Sigmoid_2,Conv1d_LSTM_model/rnn/while/lstm_cell/Tanh_1*
T0*(
_output_shapes
:         ╚
·
?Conv1d_LSTM_model/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3EConv1d_LSTM_model/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter&Conv1d_LSTM_model/rnn/while/Identity_1+Conv1d_LSTM_model/rnn/while/lstm_cell/mul_2&Conv1d_LSTM_model/rnn/while/Identity_2*>
_class4
20loc:@Conv1d_LSTM_model/rnn/while/lstm_cell/mul_2*
_output_shapes
: *
T0
┐
EConv1d_LSTM_model/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter!Conv1d_LSTM_model/rnn/TensorArray*
is_constant(*
_output_shapes
:*9

frame_name+)Conv1d_LSTM_model/rnn/while/while_context*
T0*>
_class4
20loc:@Conv1d_LSTM_model/rnn/while/lstm_cell/mul_2*
parallel_iterations 
М
#Conv1d_LSTM_model/rnn/while/add_1/yConst%^Conv1d_LSTM_model/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Ц
!Conv1d_LSTM_model/rnn/while/add_1Add&Conv1d_LSTM_model/rnn/while/Identity_1#Conv1d_LSTM_model/rnn/while/add_1/y*
T0*
_output_shapes
: 
|
)Conv1d_LSTM_model/rnn/while/NextIterationNextIterationConv1d_LSTM_model/rnn/while/add*
T0*
_output_shapes
: 
А
+Conv1d_LSTM_model/rnn/while/NextIteration_1NextIteration!Conv1d_LSTM_model/rnn/while/add_1*
_output_shapes
: *
T0
Ю
+Conv1d_LSTM_model/rnn/while/NextIteration_2NextIteration?Conv1d_LSTM_model/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Ь
+Conv1d_LSTM_model/rnn/while/NextIteration_3NextIteration+Conv1d_LSTM_model/rnn/while/lstm_cell/add_1*
T0*(
_output_shapes
:         ╚
Ь
+Conv1d_LSTM_model/rnn/while/NextIteration_4NextIteration+Conv1d_LSTM_model/rnn/while/lstm_cell/mul_2*
T0*(
_output_shapes
:         ╚
m
 Conv1d_LSTM_model/rnn/while/ExitExit"Conv1d_LSTM_model/rnn/while/Switch*
T0*
_output_shapes
: 
q
"Conv1d_LSTM_model/rnn/while/Exit_1Exit$Conv1d_LSTM_model/rnn/while/Switch_1*
_output_shapes
: *
T0
q
"Conv1d_LSTM_model/rnn/while/Exit_2Exit$Conv1d_LSTM_model/rnn/while/Switch_2*
T0*
_output_shapes
: 
Г
"Conv1d_LSTM_model/rnn/while/Exit_3Exit$Conv1d_LSTM_model/rnn/while/Switch_3*
T0*(
_output_shapes
:         ╚
Г
"Conv1d_LSTM_model/rnn/while/Exit_4Exit$Conv1d_LSTM_model/rnn/while/Switch_4*
T0*(
_output_shapes
:         ╚
т
8Conv1d_LSTM_model/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3!Conv1d_LSTM_model/rnn/TensorArray"Conv1d_LSTM_model/rnn/while/Exit_2*4
_class*
(&loc:@Conv1d_LSTM_model/rnn/TensorArray*
_output_shapes
: 
к
2Conv1d_LSTM_model/rnn/TensorArrayStack/range/startConst*4
_class*
(&loc:@Conv1d_LSTM_model/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
к
2Conv1d_LSTM_model/rnn/TensorArrayStack/range/deltaConst*4
_class*
(&loc:@Conv1d_LSTM_model/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
╛
,Conv1d_LSTM_model/rnn/TensorArrayStack/rangeRange2Conv1d_LSTM_model/rnn/TensorArrayStack/range/start8Conv1d_LSTM_model/rnn/TensorArrayStack/TensorArraySizeV32Conv1d_LSTM_model/rnn/TensorArrayStack/range/delta*#
_output_shapes
:         *

Tidx0*4
_class*
(&loc:@Conv1d_LSTM_model/rnn/TensorArray
▀
:Conv1d_LSTM_model/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3!Conv1d_LSTM_model/rnn/TensorArray,Conv1d_LSTM_model/rnn/TensorArrayStack/range"Conv1d_LSTM_model/rnn/while/Exit_2*%
element_shape:         ╚*4
_class*
(&loc:@Conv1d_LSTM_model/rnn/TensorArray*
dtype0*-
_output_shapes
:ш         ╚
h
Conv1d_LSTM_model/rnn/Const_1Const*
valueB:╚*
dtype0*
_output_shapes
:
^
Conv1d_LSTM_model/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
e
#Conv1d_LSTM_model/rnn/range_1/startConst*
_output_shapes
: *
value	B :*
dtype0
e
#Conv1d_LSTM_model/rnn/range_1/deltaConst*
_output_shapes
: *
value	B :*
dtype0
╢
Conv1d_LSTM_model/rnn/range_1Range#Conv1d_LSTM_model/rnn/range_1/startConv1d_LSTM_model/rnn/Rank_1#Conv1d_LSTM_model/rnn/range_1/delta*
_output_shapes
:*

Tidx0
x
'Conv1d_LSTM_model/rnn/concat_2/values_0Const*
_output_shapes
:*
valueB"       *
dtype0
e
#Conv1d_LSTM_model/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╤
Conv1d_LSTM_model/rnn/concat_2ConcatV2'Conv1d_LSTM_model/rnn/concat_2/values_0Conv1d_LSTM_model/rnn/range_1#Conv1d_LSTM_model/rnn/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
╧
!Conv1d_LSTM_model/rnn/transpose_1	Transpose:Conv1d_LSTM_model/rnn/TensorArrayStack/TensorArrayGatherV3Conv1d_LSTM_model/rnn/concat_2*
T0*-
_output_shapes
:         ш╚*
Tperm0
z
%Conv1d_LSTM_model/strided_slice/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
|
'Conv1d_LSTM_model/strided_slice/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
|
'Conv1d_LSTM_model/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
я
Conv1d_LSTM_model/strided_sliceStridedSlice!Conv1d_LSTM_model/rnn/transpose_1%Conv1d_LSTM_model/strided_slice/stack'Conv1d_LSTM_model/strided_slice/stack_1'Conv1d_LSTM_model/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*(
_output_shapes
:         ╚*
T0*
Index0
├
?Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform/shapeConst*1
_class'
%#loc:@Conv1d_LSTM_model/dense/kernel*
valueB"╚      *
dtype0*
_output_shapes
:
╡
=Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform/minConst*1
_class'
%#loc:@Conv1d_LSTM_model/dense/kernel*
valueB
 *W{0╛*
dtype0*
_output_shapes
: 
╡
=Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *1
_class'
%#loc:@Conv1d_LSTM_model/dense/kernel*
valueB
 *W{0>
Ь
GConv1d_LSTM_model/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform?Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	╚*

seed *
T0*1
_class'
%#loc:@Conv1d_LSTM_model/dense/kernel
Ц
=Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform/subSub=Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform/max=Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform/min*1
_class'
%#loc:@Conv1d_LSTM_model/dense/kernel*
_output_shapes
: *
T0
й
=Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform/mulMulGConv1d_LSTM_model/dense/kernel/Initializer/random_uniform/RandomUniform=Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@Conv1d_LSTM_model/dense/kernel*
_output_shapes
:	╚
Ы
9Conv1d_LSTM_model/dense/kernel/Initializer/random_uniformAdd=Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform/mul=Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@Conv1d_LSTM_model/dense/kernel*
_output_shapes
:	╚
╟
Conv1d_LSTM_model/dense/kernel
VariableV2*
shared_name *1
_class'
%#loc:@Conv1d_LSTM_model/dense/kernel*
	container *
shape:	╚*
dtype0*
_output_shapes
:	╚
Р
%Conv1d_LSTM_model/dense/kernel/AssignAssignConv1d_LSTM_model/dense/kernel9Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform*
T0*1
_class'
%#loc:@Conv1d_LSTM_model/dense/kernel*
validate_shape(*
_output_shapes
:	╚*
use_locking(
м
#Conv1d_LSTM_model/dense/kernel/readIdentityConv1d_LSTM_model/dense/kernel*
_output_shapes
:	╚*
T0*1
_class'
%#loc:@Conv1d_LSTM_model/dense/kernel
м
.Conv1d_LSTM_model/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*/
_class%
#!loc:@Conv1d_LSTM_model/dense/bias*
valueB*    
╣
Conv1d_LSTM_model/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name */
_class%
#!loc:@Conv1d_LSTM_model/dense/bias*
	container *
shape:
·
#Conv1d_LSTM_model/dense/bias/AssignAssignConv1d_LSTM_model/dense/bias.Conv1d_LSTM_model/dense/bias/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@Conv1d_LSTM_model/dense/bias*
validate_shape(*
_output_shapes
:
б
!Conv1d_LSTM_model/dense/bias/readIdentityConv1d_LSTM_model/dense/bias*
T0*/
_class%
#!loc:@Conv1d_LSTM_model/dense/bias*
_output_shapes
:
╞
Conv1d_LSTM_model/dense/MatMulMatMulConv1d_LSTM_model/strided_slice#Conv1d_LSTM_model/dense/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
╢
Conv1d_LSTM_model/dense/BiasAddBiasAddConv1d_LSTM_model/dense/MatMul!Conv1d_LSTM_model/dense/bias/read*
data_formatNHWC*'
_output_shapes
:         *
T0
R
ArgMax/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
Р
ArgMaxArgMaxConv1d_LSTM_model/dense/BiasAddArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:         
l
softmax_tensorSoftmaxConv1d_LSTM_model/dense/BiasAdd*
T0*'
_output_shapes
:         
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_6a3ec0a6790941bcb259ce472e425318/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
К
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*о
valueдBбBConv1d_LSTM_model/dense/biasBConv1d_LSTM_model/dense/kernelB$Conv1d_LSTM_model/rnn/lstm_cell/biasB&Conv1d_LSTM_model/rnn/lstm_cell/kernelBglobal_step
|
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B *
dtype0*
_output_shapes
:
а
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesConv1d_LSTM_model/dense/biasConv1d_LSTM_model/dense/kernel$Conv1d_LSTM_model/rnn/lstm_cell/bias&Conv1d_LSTM_model/rnn/lstm_cell/kernelglobal_step"/device:CPU:0*
dtypes	
2	
а
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
м
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
М
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
Й
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
Н
save/RestoreV2/tensor_namesConst"/device:CPU:0*о
valueдBбBConv1d_LSTM_model/dense/biasBConv1d_LSTM_model/dense/kernelB$Conv1d_LSTM_model/rnn/lstm_cell/biasB&Conv1d_LSTM_model/rnn/lstm_cell/kernelBglobal_step*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B *
dtype0*
_output_shapes
:
│
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
2	
┬
save/AssignAssignConv1d_LSTM_model/dense/biassave/RestoreV2*/
_class%
#!loc:@Conv1d_LSTM_model/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
╧
save/Assign_1AssignConv1d_LSTM_model/dense/kernelsave/RestoreV2:1*1
_class'
%#loc:@Conv1d_LSTM_model/dense/kernel*
validate_shape(*
_output_shapes
:	╚*
use_locking(*
T0
╫
save/Assign_2Assign$Conv1d_LSTM_model/rnn/lstm_cell/biassave/RestoreV2:2*
use_locking(*
T0*7
_class-
+)loc:@Conv1d_LSTM_model/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:а
р
save/Assign_3Assign&Conv1d_LSTM_model/rnn/lstm_cell/kernelsave/RestoreV2:3*
use_locking(*
T0*9
_class/
-+loc:@Conv1d_LSTM_model/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
╩а
а
save/Assign_4Assignglobal_stepsave/RestoreV2:4*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
h
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4
-
save/restore_allNoOp^save/restore_shard

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
R
save_1/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
Ж
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_6ece2ba7790d4d75a0c36e54e6b24021/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
Ф
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
М
save_1/SaveV2/tensor_namesConst"/device:CPU:0*о
valueдBбBConv1d_LSTM_model/dense/biasBConv1d_LSTM_model/dense/kernelB$Conv1d_LSTM_model/rnn/lstm_cell/biasB&Conv1d_LSTM_model/rnn/lstm_cell/kernelBglobal_step*
dtype0*
_output_shapes
:
~
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B *
dtype0*
_output_shapes
:
и
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesConv1d_LSTM_model/dense/biasConv1d_LSTM_model/dense/kernel$Conv1d_LSTM_model/rnn/lstm_cell/bias&Conv1d_LSTM_model/rnn/lstm_cell/kernelglobal_step"/device:CPU:0*
dtypes	
2	
и
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
▓
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
Т
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
С
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
П
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*о
valueдBбBConv1d_LSTM_model/dense/biasBConv1d_LSTM_model/dense/kernelB$Conv1d_LSTM_model/rnn/lstm_cell/biasB&Conv1d_LSTM_model/rnn/lstm_cell/kernelBglobal_step*
dtype0*
_output_shapes
:
Б
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueBB B B B B *
dtype0
╗
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
2	
╞
save_1/AssignAssignConv1d_LSTM_model/dense/biassave_1/RestoreV2*
use_locking(*
T0*/
_class%
#!loc:@Conv1d_LSTM_model/dense/bias*
validate_shape(*
_output_shapes
:
╙
save_1/Assign_1AssignConv1d_LSTM_model/dense/kernelsave_1/RestoreV2:1*
use_locking(*
T0*1
_class'
%#loc:@Conv1d_LSTM_model/dense/kernel*
validate_shape(*
_output_shapes
:	╚
█
save_1/Assign_2Assign$Conv1d_LSTM_model/rnn/lstm_cell/biassave_1/RestoreV2:2*
use_locking(*
T0*7
_class-
+)loc:@Conv1d_LSTM_model/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:а
ф
save_1/Assign_3Assign&Conv1d_LSTM_model/rnn/lstm_cell/kernelsave_1/RestoreV2:3*
use_locking(*
T0*9
_class/
-+loc:@Conv1d_LSTM_model/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
╩а
д
save_1/Assign_4Assignglobal_stepsave_1/RestoreV2:4*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
t
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8" 
legacy_init_op


group_deps"∙
trainable_variablesс▐
═
(Conv1d_LSTM_model/rnn/lstm_cell/kernel:0-Conv1d_LSTM_model/rnn/lstm_cell/kernel/Assign-Conv1d_LSTM_model/rnn/lstm_cell/kernel/read:02CConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform:0
╝
&Conv1d_LSTM_model/rnn/lstm_cell/bias:0+Conv1d_LSTM_model/rnn/lstm_cell/bias/Assign+Conv1d_LSTM_model/rnn/lstm_cell/bias/read:028Conv1d_LSTM_model/rnn/lstm_cell/bias/Initializer/zeros:0
н
 Conv1d_LSTM_model/dense/kernel:0%Conv1d_LSTM_model/dense/kernel/Assign%Conv1d_LSTM_model/dense/kernel/read:02;Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform:0
Ь
Conv1d_LSTM_model/dense/bias:0#Conv1d_LSTM_model/dense/bias/Assign#Conv1d_LSTM_model/dense/bias/read:020Conv1d_LSTM_model/dense/bias/Initializer/zeros:0"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"╒'
while_context├'└'
╜'
)Conv1d_LSTM_model/rnn/while/while_context *&Conv1d_LSTM_model/rnn/while/LoopCond:02#Conv1d_LSTM_model/rnn/while/Merge:0:&Conv1d_LSTM_model/rnn/while/Identity:0B"Conv1d_LSTM_model/rnn/while/Exit:0B$Conv1d_LSTM_model/rnn/while/Exit_1:0B$Conv1d_LSTM_model/rnn/while/Exit_2:0B$Conv1d_LSTM_model/rnn/while/Exit_3:0B$Conv1d_LSTM_model/rnn/while/Exit_4:0JЁ"
Conv1d_LSTM_model/rnn/Minimum:0
#Conv1d_LSTM_model/rnn/TensorArray:0
RConv1d_LSTM_model/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
%Conv1d_LSTM_model/rnn/TensorArray_1:0
+Conv1d_LSTM_model/rnn/lstm_cell/bias/read:0
-Conv1d_LSTM_model/rnn/lstm_cell/kernel/read:0
'Conv1d_LSTM_model/rnn/strided_slice_1:0
#Conv1d_LSTM_model/rnn/while/Enter:0
%Conv1d_LSTM_model/rnn/while/Enter_1:0
%Conv1d_LSTM_model/rnn/while/Enter_2:0
%Conv1d_LSTM_model/rnn/while/Enter_3:0
%Conv1d_LSTM_model/rnn/while/Enter_4:0
"Conv1d_LSTM_model/rnn/while/Exit:0
$Conv1d_LSTM_model/rnn/while/Exit_1:0
$Conv1d_LSTM_model/rnn/while/Exit_2:0
$Conv1d_LSTM_model/rnn/while/Exit_3:0
$Conv1d_LSTM_model/rnn/while/Exit_4:0
&Conv1d_LSTM_model/rnn/while/Identity:0
(Conv1d_LSTM_model/rnn/while/Identity_1:0
(Conv1d_LSTM_model/rnn/while/Identity_2:0
(Conv1d_LSTM_model/rnn/while/Identity_3:0
(Conv1d_LSTM_model/rnn/while/Identity_4:0
(Conv1d_LSTM_model/rnn/while/Less/Enter:0
"Conv1d_LSTM_model/rnn/while/Less:0
*Conv1d_LSTM_model/rnn/while/Less_1/Enter:0
$Conv1d_LSTM_model/rnn/while/Less_1:0
(Conv1d_LSTM_model/rnn/while/LogicalAnd:0
&Conv1d_LSTM_model/rnn/while/LoopCond:0
#Conv1d_LSTM_model/rnn/while/Merge:0
#Conv1d_LSTM_model/rnn/while/Merge:1
%Conv1d_LSTM_model/rnn/while/Merge_1:0
%Conv1d_LSTM_model/rnn/while/Merge_1:1
%Conv1d_LSTM_model/rnn/while/Merge_2:0
%Conv1d_LSTM_model/rnn/while/Merge_2:1
%Conv1d_LSTM_model/rnn/while/Merge_3:0
%Conv1d_LSTM_model/rnn/while/Merge_3:1
%Conv1d_LSTM_model/rnn/while/Merge_4:0
%Conv1d_LSTM_model/rnn/while/Merge_4:1
+Conv1d_LSTM_model/rnn/while/NextIteration:0
-Conv1d_LSTM_model/rnn/while/NextIteration_1:0
-Conv1d_LSTM_model/rnn/while/NextIteration_2:0
-Conv1d_LSTM_model/rnn/while/NextIteration_3:0
-Conv1d_LSTM_model/rnn/while/NextIteration_4:0
$Conv1d_LSTM_model/rnn/while/Switch:0
$Conv1d_LSTM_model/rnn/while/Switch:1
&Conv1d_LSTM_model/rnn/while/Switch_1:0
&Conv1d_LSTM_model/rnn/while/Switch_1:1
&Conv1d_LSTM_model/rnn/while/Switch_2:0
&Conv1d_LSTM_model/rnn/while/Switch_2:1
&Conv1d_LSTM_model/rnn/while/Switch_3:0
&Conv1d_LSTM_model/rnn/while/Switch_3:1
&Conv1d_LSTM_model/rnn/while/Switch_4:0
&Conv1d_LSTM_model/rnn/while/Switch_4:1
5Conv1d_LSTM_model/rnn/while/TensorArrayReadV3/Enter:0
7Conv1d_LSTM_model/rnn/while/TensorArrayReadV3/Enter_1:0
/Conv1d_LSTM_model/rnn/while/TensorArrayReadV3:0
GConv1d_LSTM_model/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
AConv1d_LSTM_model/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
#Conv1d_LSTM_model/rnn/while/add/y:0
!Conv1d_LSTM_model/rnn/while/add:0
%Conv1d_LSTM_model/rnn/while/add_1/y:0
#Conv1d_LSTM_model/rnn/while/add_1:0
5Conv1d_LSTM_model/rnn/while/lstm_cell/BiasAdd/Enter:0
/Conv1d_LSTM_model/rnn/while/lstm_cell/BiasAdd:0
-Conv1d_LSTM_model/rnn/while/lstm_cell/Const:0
4Conv1d_LSTM_model/rnn/while/lstm_cell/MatMul/Enter:0
.Conv1d_LSTM_model/rnn/while/lstm_cell/MatMul:0
/Conv1d_LSTM_model/rnn/while/lstm_cell/Sigmoid:0
1Conv1d_LSTM_model/rnn/while/lstm_cell/Sigmoid_1:0
1Conv1d_LSTM_model/rnn/while/lstm_cell/Sigmoid_2:0
,Conv1d_LSTM_model/rnn/while/lstm_cell/Tanh:0
.Conv1d_LSTM_model/rnn/while/lstm_cell/Tanh_1:0
-Conv1d_LSTM_model/rnn/while/lstm_cell/add/y:0
+Conv1d_LSTM_model/rnn/while/lstm_cell/add:0
-Conv1d_LSTM_model/rnn/while/lstm_cell/add_1:0
3Conv1d_LSTM_model/rnn/while/lstm_cell/concat/axis:0
.Conv1d_LSTM_model/rnn/while/lstm_cell/concat:0
+Conv1d_LSTM_model/rnn/while/lstm_cell/mul:0
-Conv1d_LSTM_model/rnn/while/lstm_cell/mul_1:0
-Conv1d_LSTM_model/rnn/while/lstm_cell/mul_2:0
7Conv1d_LSTM_model/rnn/while/lstm_cell/split/split_dim:0
-Conv1d_LSTM_model/rnn/while/lstm_cell/split:0
-Conv1d_LSTM_model/rnn/while/lstm_cell/split:1
-Conv1d_LSTM_model/rnn/while/lstm_cell/split:2
-Conv1d_LSTM_model/rnn/while/lstm_cell/split:3S
'Conv1d_LSTM_model/rnn/strided_slice_1:0(Conv1d_LSTM_model/rnn/while/Less/Enter:0^
%Conv1d_LSTM_model/rnn/TensorArray_1:05Conv1d_LSTM_model/rnn/while/TensorArrayReadV3/Enter:0n
#Conv1d_LSTM_model/rnn/TensorArray:0GConv1d_LSTM_model/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0M
Conv1d_LSTM_model/rnn/Minimum:0*Conv1d_LSTM_model/rnn/while/Less_1/Enter:0e
-Conv1d_LSTM_model/rnn/lstm_cell/kernel/read:04Conv1d_LSTM_model/rnn/while/lstm_cell/MatMul/Enter:0d
+Conv1d_LSTM_model/rnn/lstm_cell/bias/read:05Conv1d_LSTM_model/rnn/while/lstm_cell/BiasAdd/Enter:0Н
RConv1d_LSTM_model/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:07Conv1d_LSTM_model/rnn/while/TensorArrayReadV3/Enter_1:0R#Conv1d_LSTM_model/rnn/while/Enter:0R%Conv1d_LSTM_model/rnn/while/Enter_1:0R%Conv1d_LSTM_model/rnn/while/Enter_2:0R%Conv1d_LSTM_model/rnn/while/Enter_3:0R%Conv1d_LSTM_model/rnn/while/Enter_4:0Z'Conv1d_LSTM_model/rnn/strided_slice_1:0"╔
	variables╗╕
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
═
(Conv1d_LSTM_model/rnn/lstm_cell/kernel:0-Conv1d_LSTM_model/rnn/lstm_cell/kernel/Assign-Conv1d_LSTM_model/rnn/lstm_cell/kernel/read:02CConv1d_LSTM_model/rnn/lstm_cell/kernel/Initializer/random_uniform:0
╝
&Conv1d_LSTM_model/rnn/lstm_cell/bias:0+Conv1d_LSTM_model/rnn/lstm_cell/bias/Assign+Conv1d_LSTM_model/rnn/lstm_cell/bias/read:028Conv1d_LSTM_model/rnn/lstm_cell/bias/Initializer/zeros:0
н
 Conv1d_LSTM_model/dense/kernel:0%Conv1d_LSTM_model/dense/kernel/Assign%Conv1d_LSTM_model/dense/kernel/read:02;Conv1d_LSTM_model/dense/kernel/Initializer/random_uniform:0
Ь
Conv1d_LSTM_model/dense/bias:0#Conv1d_LSTM_model/dense/bias/Assign#Conv1d_LSTM_model/dense/bias/read:020Conv1d_LSTM_model/dense/bias/Initializer/zeros:0*╕
predictм
,
input#
input:0         ш&
classes
ArgMax:0	         8
probabilities'
softmax_tensor:0         tensorflow/serving/predict*└
serving_defaultм
,
input#
input:0         ш8
probabilities'
softmax_tensor:0         &
classes
ArgMax:0	         tensorflow/serving/predict