¾¬
ó
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
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

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
¾
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
executor_typestring 
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

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02unknown8Âç

r
gcn_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namegcn_layer/w
k
gcn_layer/w/Read/ReadVariableOpReadVariableOpgcn_layer/w*
_output_shapes

:*
dtype0
v
gcn_layer_1/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namegcn_layer_1/w
o
!gcn_layer_1/w/Read/ReadVariableOpReadVariableOpgcn_layer_1/w*
_output_shapes

:*
dtype0
v
gcn_layer_2/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namegcn_layer_2/w
o
!gcn_layer_2/w/Read/ReadVariableOpReadVariableOpgcn_layer_2/w*
_output_shapes

:*
dtype0
v
gcn_layer_3/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namegcn_layer_3/w
o
!gcn_layer_3/w/Read/ReadVariableOpReadVariableOpgcn_layer_3/w*
_output_shapes

:*
dtype0
v
gcn_layer_4/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namegcn_layer_4/w
o
!gcn_layer_4/w/Read/ReadVariableOpReadVariableOpgcn_layer_4/w*
_output_shapes

:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:
*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:
*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:
*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/gcn_layer/w/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_nameAdam/gcn_layer/w/m
y
&Adam/gcn_layer/w/m/Read/ReadVariableOpReadVariableOpAdam/gcn_layer/w/m*
_output_shapes

:*
dtype0

Adam/gcn_layer_1/w/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/gcn_layer_1/w/m
}
(Adam/gcn_layer_1/w/m/Read/ReadVariableOpReadVariableOpAdam/gcn_layer_1/w/m*
_output_shapes

:*
dtype0

Adam/gcn_layer_2/w/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/gcn_layer_2/w/m
}
(Adam/gcn_layer_2/w/m/Read/ReadVariableOpReadVariableOpAdam/gcn_layer_2/w/m*
_output_shapes

:*
dtype0

Adam/gcn_layer_3/w/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/gcn_layer_3/w/m
}
(Adam/gcn_layer_3/w/m/Read/ReadVariableOpReadVariableOpAdam/gcn_layer_3/w/m*
_output_shapes

:*
dtype0

Adam/gcn_layer_4/w/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/gcn_layer_4/w/m
}
(Adam/gcn_layer_4/w/m/Read/ReadVariableOpReadVariableOpAdam/gcn_layer_4/w/m*
_output_shapes

:*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:
*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:
*
dtype0

Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:
*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0

Adam/gcn_layer/w/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_nameAdam/gcn_layer/w/v
y
&Adam/gcn_layer/w/v/Read/ReadVariableOpReadVariableOpAdam/gcn_layer/w/v*
_output_shapes

:*
dtype0

Adam/gcn_layer_1/w/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/gcn_layer_1/w/v
}
(Adam/gcn_layer_1/w/v/Read/ReadVariableOpReadVariableOpAdam/gcn_layer_1/w/v*
_output_shapes

:*
dtype0

Adam/gcn_layer_2/w/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/gcn_layer_2/w/v
}
(Adam/gcn_layer_2/w/v/Read/ReadVariableOpReadVariableOpAdam/gcn_layer_2/w/v*
_output_shapes

:*
dtype0

Adam/gcn_layer_3/w/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/gcn_layer_3/w/v
}
(Adam/gcn_layer_3/w/v/Read/ReadVariableOpReadVariableOpAdam/gcn_layer_3/w/v*
_output_shapes

:*
dtype0

Adam/gcn_layer_4/w/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/gcn_layer_4/w/v
}
(Adam/gcn_layer_4/w/v/Read/ReadVariableOpReadVariableOpAdam/gcn_layer_4/w/v*
_output_shapes

:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:
*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:
*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:
*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
A
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ï@
valueÅ@BÂ@ B»@
Ñ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
 
Y
w
regularization_losses
trainable_variables
	variables
	keras_api
Y
w
regularization_losses
trainable_variables
	variables
	keras_api
Y
w
regularization_losses
trainable_variables
	variables
 	keras_api
Y
!w
"regularization_losses
#trainable_variables
$	variables
%	keras_api
Y
&w
'regularization_losses
(trainable_variables
)	variables
*	keras_api
R
+regularization_losses
,trainable_variables
-	variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
h

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
h

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api

Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratem}m~m!m&m/m0m5m6m;m<mvvv!v&v/v0v5v6v;v<v
 
N
0
1
2
!3
&4
/5
06
57
68
;9
<10
N
0
1
2
!3
&4
/5
06
57
68
;9
<10
­
Fnon_trainable_variables
regularization_losses
trainable_variables
Glayer_regularization_losses
Hlayer_metrics
	variables
Imetrics

Jlayers
 
RP
VARIABLE_VALUEgcn_layer/w1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
Knon_trainable_variables
regularization_losses
Llayer_metrics
Mlayer_regularization_losses
trainable_variables
	variables
Nmetrics

Olayers
TR
VARIABLE_VALUEgcn_layer_1/w1layer_with_weights-1/w/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
Pnon_trainable_variables
regularization_losses
Qlayer_metrics
Rlayer_regularization_losses
trainable_variables
	variables
Smetrics

Tlayers
TR
VARIABLE_VALUEgcn_layer_2/w1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
Unon_trainable_variables
regularization_losses
Vlayer_metrics
Wlayer_regularization_losses
trainable_variables
	variables
Xmetrics

Ylayers
TR
VARIABLE_VALUEgcn_layer_3/w1layer_with_weights-3/w/.ATTRIBUTES/VARIABLE_VALUE
 

!0

!0
­
Znon_trainable_variables
"regularization_losses
[layer_metrics
\layer_regularization_losses
#trainable_variables
$	variables
]metrics

^layers
TR
VARIABLE_VALUEgcn_layer_4/w1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUE
 

&0

&0
­
_non_trainable_variables
'regularization_losses
`layer_metrics
alayer_regularization_losses
(trainable_variables
)	variables
bmetrics

clayers
 
 
 
­
dnon_trainable_variables
+regularization_losses
elayer_metrics
flayer_regularization_losses
,trainable_variables
-	variables
gmetrics

hlayers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
­
inon_trainable_variables
1regularization_losses
jlayer_metrics
klayer_regularization_losses
2trainable_variables
3	variables
lmetrics

mlayers
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
­
nnon_trainable_variables
7regularization_losses
olayer_metrics
player_regularization_losses
8trainable_variables
9	variables
qmetrics

rlayers
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
­
snon_trainable_variables
=regularization_losses
tlayer_metrics
ulayer_regularization_losses
>trainable_variables
?	variables
vmetrics

wlayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

x0
N
0
1
2
3
4
5
6
7
	8

9
10
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
4
	ytotal
	zcount
{	variables
|	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

y0
z1

{	variables
us
VARIABLE_VALUEAdam/gcn_layer/w/mMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/gcn_layer_1/w/mMlayer_with_weights-1/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/gcn_layer_2/w/mMlayer_with_weights-2/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/gcn_layer_3/w/mMlayer_with_weights-3/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/gcn_layer_4/w/mMlayer_with_weights-4/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/gcn_layer/w/vMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/gcn_layer_1/w/vMlayer_with_weights-1/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/gcn_layer_2/w/vMlayer_with_weights-2/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/gcn_layer_3/w/vMlayer_with_weights-3/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/gcn_layer_4/w/vMlayer_with_weights-4/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_2Placeholder*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*)
shape :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦
serving_default_input_3Placeholder*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*2
shape):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2serving_default_input_3gcn_layer/wgcn_layer_1/wgcn_layer_2/wgcn_layer_3/wgcn_layer_4/wdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_148012
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¹
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamegcn_layer/w/Read/ReadVariableOp!gcn_layer_1/w/Read/ReadVariableOp!gcn_layer_2/w/Read/ReadVariableOp!gcn_layer_3/w/Read/ReadVariableOp!gcn_layer_4/w/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp&Adam/gcn_layer/w/m/Read/ReadVariableOp(Adam/gcn_layer_1/w/m/Read/ReadVariableOp(Adam/gcn_layer_2/w/m/Read/ReadVariableOp(Adam/gcn_layer_3/w/m/Read/ReadVariableOp(Adam/gcn_layer_4/w/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp&Adam/gcn_layer/w/v/Read/ReadVariableOp(Adam/gcn_layer_1/w/v/Read/ReadVariableOp(Adam/gcn_layer_2/w/v/Read/ReadVariableOp(Adam/gcn_layer_3/w/v/Read/ReadVariableOp(Adam/gcn_layer_4/w/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOpConst*5
Tin.
,2*	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_148571

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegcn_layer/wgcn_layer_1/wgcn_layer_2/wgcn_layer_3/wgcn_layer_4/wdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/gcn_layer/w/mAdam/gcn_layer_1/w/mAdam/gcn_layer_2/w/mAdam/gcn_layer_3/w/mAdam/gcn_layer_4/w/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/gcn_layer/w/vAdam/gcn_layer_1/w/vAdam/gcn_layer_2/w/vAdam/gcn_layer_3/w/vAdam/gcn_layer_4/w/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/v*4
Tin-
+2)*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_148701î¼	

ô
C__inference_dense_4_layer_call_and_return_conditional_losses_148427

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

T
(__inference_GRLayer_layer_call_fn_148360
inputs_0
inputs_1
identityÎ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_GRLayer_layer_call_and_return_conditional_losses_1475982
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¾
¢
(__inference_model_2_layer_call_fn_148068
inputs_0
inputs_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:

	unknown_7:

	unknown_8:

	unknown_9:
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1478412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ï

(__inference_dense_2_layer_call_fn_148376

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1476112
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ«

"__inference__traced_restore_148701
file_prefix.
assignvariableop_gcn_layer_w:2
 assignvariableop_1_gcn_layer_1_w:2
 assignvariableop_2_gcn_layer_2_w:2
 assignvariableop_3_gcn_layer_3_w:2
 assignvariableop_4_gcn_layer_4_w:3
!assignvariableop_5_dense_2_kernel:-
assignvariableop_6_dense_2_bias:3
!assignvariableop_7_dense_3_kernel:
-
assignvariableop_8_dense_3_bias:
3
!assignvariableop_9_dense_4_kernel:
.
 assignvariableop_10_dense_4_bias:'
assignvariableop_11_adam_iter:	 )
assignvariableop_12_adam_beta_1: )
assignvariableop_13_adam_beta_2: (
assignvariableop_14_adam_decay: 0
&assignvariableop_15_adam_learning_rate: #
assignvariableop_16_total: #
assignvariableop_17_count: 8
&assignvariableop_18_adam_gcn_layer_w_m::
(assignvariableop_19_adam_gcn_layer_1_w_m::
(assignvariableop_20_adam_gcn_layer_2_w_m::
(assignvariableop_21_adam_gcn_layer_3_w_m::
(assignvariableop_22_adam_gcn_layer_4_w_m:;
)assignvariableop_23_adam_dense_2_kernel_m:5
'assignvariableop_24_adam_dense_2_bias_m:;
)assignvariableop_25_adam_dense_3_kernel_m:
5
'assignvariableop_26_adam_dense_3_bias_m:
;
)assignvariableop_27_adam_dense_4_kernel_m:
5
'assignvariableop_28_adam_dense_4_bias_m:8
&assignvariableop_29_adam_gcn_layer_w_v::
(assignvariableop_30_adam_gcn_layer_1_w_v::
(assignvariableop_31_adam_gcn_layer_2_w_v::
(assignvariableop_32_adam_gcn_layer_3_w_v::
(assignvariableop_33_adam_gcn_layer_4_w_v:;
)assignvariableop_34_adam_dense_2_kernel_v:5
'assignvariableop_35_adam_dense_2_bias_v:;
)assignvariableop_36_adam_dense_3_kernel_v:
5
'assignvariableop_37_adam_dense_3_bias_v:
;
)assignvariableop_38_adam_dense_4_kernel_v:
5
'assignvariableop_39_adam_dense_4_bias_v:
identity_41¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9»
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*Ç
value½Bº)B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-1/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-3/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-1/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-3/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-4/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-1/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-3/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-4/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesà
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesû
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_gcn_layer_wIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_gcn_layer_1_wIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¥
AssignVariableOp_2AssignVariableOp assignvariableop_2_gcn_layer_2_wIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_gcn_layer_3_wIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¥
AssignVariableOp_4AssignVariableOp assignvariableop_4_gcn_layer_4_wIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¤
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¤
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¨
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_4_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11¥
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12§
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¦
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15®
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¡
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¡
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18®
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_gcn_layer_w_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19°
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_gcn_layer_1_w_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_gcn_layer_2_w_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21°
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_gcn_layer_3_w_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22°
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_gcn_layer_4_w_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23±
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¯
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25±
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¯
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27±
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¯
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29®
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_gcn_layer_w_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30°
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_gcn_layer_1_w_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31°
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_gcn_layer_2_w_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32°
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_gcn_layer_3_w_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33°
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_gcn_layer_4_w_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_2_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¯
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_2_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36±
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_3_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¯
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_3_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38±
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_4_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¯
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_4_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_399
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÎ
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_40f
Identity_41IdentityIdentity_40:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_41¶
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_41Identity_41:output:0*e
_input_shapesT
R: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
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
ï

(__inference_dense_3_layer_call_fn_148396

inputs
unknown:

	unknown_0:

identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1476282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Þ
G__inference_gcn_layer_1_layer_call_and_return_conditional_losses_148276
inputs_0
inputs_17
%einsum_einsum_readvariableop_resource:
identity

identity_1¢einsum/Einsum/ReadVariableOpy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indicesv
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Sum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xz
truedivRealDivtruediv/x:output:0Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
truediv¢
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02
einsum/Einsum/ReadVariableOp»
einsum/EinsumEinsum$einsum/Einsum/ReadVariableOp:value:0inputs_0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
einsum/Einsum²
einsum/Einsum_1Einsumeinsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
einsum/Einsum_1¶
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
einsum/Einsum_2m
ReluRelueinsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1m
NoOpNoOp^einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¢
Ü
G__inference_gcn_layer_2_layer_call_and_return_conditional_losses_147544

inputs
inputs_17
%einsum_einsum_readvariableop_resource:
identity

identity_1¢einsum/Einsum/ReadVariableOpy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indicesv
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Sum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xz
truedivRealDivtruediv/x:output:0Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
truediv¢
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02
einsum/Einsum/ReadVariableOp¹
einsum/EinsumEinsum$einsum/Einsum/ReadVariableOp:value:0inputs*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
einsum/Einsum²
einsum/Einsum_1Einsumeinsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
einsum/Einsum_1¶
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
einsum/Einsum_2m
ReluRelueinsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1m
NoOpNoOp^einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
 
,__inference_gcn_layer_1_layer_call_fn_148260
inputs_0
inputs_1
unknown:
identity

identity_1¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_1_layer_call_and_return_conditional_losses_1475232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ª
 
,__inference_gcn_layer_2_layer_call_fn_148286
inputs_0
inputs_1
unknown:
identity

identity_1¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_2_layer_call_and_return_conditional_losses_1475442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1

ô
C__inference_dense_3_layer_call_and_return_conditional_losses_147628

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
 
,__inference_gcn_layer_4_layer_call_fn_148338
inputs_0
inputs_1
unknown:
identity

identity_1¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_4_layer_call_and_return_conditional_losses_1475862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¢
Ü
G__inference_gcn_layer_4_layer_call_and_return_conditional_losses_147586

inputs
inputs_17
%einsum_einsum_readvariableop_resource:
identity

identity_1¢einsum/Einsum/ReadVariableOpy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indicesv
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Sum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xz
truedivRealDivtruediv/x:output:0Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
truediv¢
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02
einsum/Einsum/ReadVariableOp¹
einsum/EinsumEinsum$einsum/Einsum/ReadVariableOp:value:0inputs*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
einsum/Einsum²
einsum/Einsum_1Einsumeinsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
einsum/Einsum_1¶
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
einsum/Einsum_2m
ReluRelueinsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1m
NoOpNoOp^einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


$__inference_signature_wrapper_148012
input_2
input_3
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:

	unknown_7:

	unknown_8:

	unknown_9:
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinput_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_1474772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
¾
¢
(__inference_model_2_layer_call_fn_148040
inputs_0
inputs_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:

	unknown_7:

	unknown_8:

	unknown_9:
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1476522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¨
Ü
E__inference_gcn_layer_layer_call_and_return_conditional_losses_148250
inputs_0
inputs_17
%einsum_einsum_readvariableop_resource:
identity

identity_1¢einsum/Einsum/ReadVariableOpy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indicesv
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Sum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xz
truedivRealDivtruediv/x:output:0Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
truediv¢
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02
einsum/Einsum/ReadVariableOp»
einsum/EinsumEinsum$einsum/Einsum/ReadVariableOp:value:0inputs_0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
einsum/Einsum²
einsum/Einsum_1Einsumeinsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
einsum/Einsum_1¶
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
einsum/Einsum_2m
ReluRelueinsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1m
NoOpNoOp^einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
 
Ú
E__inference_gcn_layer_layer_call_and_return_conditional_losses_147502

inputs
inputs_17
%einsum_einsum_readvariableop_resource:
identity

identity_1¢einsum/Einsum/ReadVariableOpy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indicesv
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Sum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xz
truedivRealDivtruediv/x:output:0Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
truediv¢
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02
einsum/Einsum/ReadVariableOp¹
einsum/EinsumEinsum$einsum/Einsum/ReadVariableOp:value:0inputs*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
einsum/Einsum²
einsum/Einsum_1Einsumeinsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
einsum/Einsum_1¶
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
einsum/Einsum_2m
ReluRelueinsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1m
NoOpNoOp^einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
 
,__inference_gcn_layer_3_layer_call_fn_148312
inputs_0
inputs_1
unknown:
identity

identity_1¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_3_layer_call_and_return_conditional_losses_1475652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ª
Þ
G__inference_gcn_layer_4_layer_call_and_return_conditional_losses_148354
inputs_0
inputs_17
%einsum_einsum_readvariableop_resource:
identity

identity_1¢einsum/Einsum/ReadVariableOpy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indicesv
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Sum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xz
truedivRealDivtruediv/x:output:0Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
truediv¢
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02
einsum/Einsum/ReadVariableOp»
einsum/EinsumEinsum$einsum/Einsum/ReadVariableOp:value:0inputs_0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
einsum/Einsum²
einsum/Einsum_1Einsumeinsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
einsum/Einsum_1¶
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
einsum/Einsum_2m
ReluRelueinsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1m
NoOpNoOp^einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1

ô
C__inference_dense_2_layer_call_and_return_conditional_losses_147611

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã3

C__inference_model_2_layer_call_and_return_conditional_losses_147652

inputs
inputs_1"
gcn_layer_147503:$
gcn_layer_1_147524:$
gcn_layer_2_147545:$
gcn_layer_3_147566:$
gcn_layer_4_147587: 
dense_2_147612:
dense_2_147614: 
dense_3_147629:

dense_3_147631:
 
dense_4_147646:

dense_4_147648:
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢!gcn_layer/StatefulPartitionedCall¢#gcn_layer_1/StatefulPartitionedCall¢#gcn_layer_2/StatefulPartitionedCall¢#gcn_layer_3/StatefulPartitionedCall¢#gcn_layer_4/StatefulPartitionedCallÇ
!gcn_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1gcn_layer_147503*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gcn_layer_layer_call_and_return_conditional_losses_1475022#
!gcn_layer/StatefulPartitionedCall
#gcn_layer_1/StatefulPartitionedCallStatefulPartitionedCall*gcn_layer/StatefulPartitionedCall:output:0*gcn_layer/StatefulPartitionedCall:output:1gcn_layer_1_147524*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_1_layer_call_and_return_conditional_losses_1475232%
#gcn_layer_1/StatefulPartitionedCall
#gcn_layer_2/StatefulPartitionedCallStatefulPartitionedCall,gcn_layer_1/StatefulPartitionedCall:output:0,gcn_layer_1/StatefulPartitionedCall:output:1gcn_layer_2_147545*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_2_layer_call_and_return_conditional_losses_1475442%
#gcn_layer_2/StatefulPartitionedCall
#gcn_layer_3/StatefulPartitionedCallStatefulPartitionedCall,gcn_layer_2/StatefulPartitionedCall:output:0,gcn_layer_2/StatefulPartitionedCall:output:1gcn_layer_3_147566*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_3_layer_call_and_return_conditional_losses_1475652%
#gcn_layer_3/StatefulPartitionedCall
#gcn_layer_4/StatefulPartitionedCallStatefulPartitionedCall,gcn_layer_3/StatefulPartitionedCall:output:0,gcn_layer_3/StatefulPartitionedCall:output:1gcn_layer_4_147587*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_4_layer_call_and_return_conditional_losses_1475862%
#gcn_layer_4/StatefulPartitionedCall¦
GRLayer/PartitionedCallPartitionedCall,gcn_layer_4/StatefulPartitionedCall:output:0,gcn_layer_4/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_GRLayer_layer_call_and_return_conditional_losses_1475982
GRLayer/PartitionedCall©
dense_2/StatefulPartitionedCallStatefulPartitionedCall GRLayer/PartitionedCall:output:0dense_2_147612dense_2_147614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1476112!
dense_2/StatefulPartitionedCall±
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_147629dense_3_147631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1476282!
dense_3/StatefulPartitionedCall±
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_147646dense_4_147648*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1476452!
dense_4/StatefulPartitionedCall
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityð
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^gcn_layer/StatefulPartitionedCall$^gcn_layer_1/StatefulPartitionedCall$^gcn_layer_2/StatefulPartitionedCall$^gcn_layer_3/StatefulPartitionedCall$^gcn_layer_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!gcn_layer/StatefulPartitionedCall!gcn_layer/StatefulPartitionedCall2J
#gcn_layer_1/StatefulPartitionedCall#gcn_layer_1/StatefulPartitionedCall2J
#gcn_layer_2/StatefulPartitionedCall#gcn_layer_2/StatefulPartitionedCall2J
#gcn_layer_3/StatefulPartitionedCall#gcn_layer_3/StatefulPartitionedCall2J
#gcn_layer_4/StatefulPartitionedCall#gcn_layer_4/StatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
m
C__inference_GRLayer_layer_call_and_return_conditional_losses_147598

inputs
inputs_1
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
 
(__inference_model_2_layer_call_fn_147894
input_2
input_3
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:

	unknown_7:

	unknown_8:

	unknown_9:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinput_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1478412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
ª
Þ
G__inference_gcn_layer_3_layer_call_and_return_conditional_losses_148328
inputs_0
inputs_17
%einsum_einsum_readvariableop_resource:
identity

identity_1¢einsum/Einsum/ReadVariableOpy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indicesv
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Sum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xz
truedivRealDivtruediv/x:output:0Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
truediv¢
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02
einsum/Einsum/ReadVariableOp»
einsum/EinsumEinsum$einsum/Einsum/ReadVariableOp:value:0inputs_0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
einsum/Einsum²
einsum/Einsum_1Einsumeinsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
einsum/Einsum_1¶
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
einsum/Einsum_2m
ReluRelueinsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1m
NoOpNoOp^einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ï

(__inference_dense_4_layer_call_fn_148416

inputs
unknown:

	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1476452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

ô
C__inference_dense_3_layer_call_and_return_conditional_losses_148407

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

*__inference_gcn_layer_layer_call_fn_148234
inputs_0
inputs_1
unknown:
identity

identity_1¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gcn_layer_layer_call_and_return_conditional_losses_1475022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¸
 
(__inference_model_2_layer_call_fn_147677
input_2
input_3
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:

	unknown_7:

	unknown_8:

	unknown_9:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinput_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1476522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
ÃT

__inference__traced_save_148571
file_prefix*
&savev2_gcn_layer_w_read_readvariableop,
(savev2_gcn_layer_1_w_read_readvariableop,
(savev2_gcn_layer_2_w_read_readvariableop,
(savev2_gcn_layer_3_w_read_readvariableop,
(savev2_gcn_layer_4_w_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop1
-savev2_adam_gcn_layer_w_m_read_readvariableop3
/savev2_adam_gcn_layer_1_w_m_read_readvariableop3
/savev2_adam_gcn_layer_2_w_m_read_readvariableop3
/savev2_adam_gcn_layer_3_w_m_read_readvariableop3
/savev2_adam_gcn_layer_4_w_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop1
-savev2_adam_gcn_layer_w_v_read_readvariableop3
/savev2_adam_gcn_layer_1_w_v_read_readvariableop3
/savev2_adam_gcn_layer_2_w_v_read_readvariableop3
/savev2_adam_gcn_layer_3_w_v_read_readvariableop3
/savev2_adam_gcn_layer_4_w_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameµ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*Ç
value½Bº)B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-1/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-3/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-1/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-3/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-4/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-1/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-3/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-4/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÚ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesð
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_gcn_layer_w_read_readvariableop(savev2_gcn_layer_1_w_read_readvariableop(savev2_gcn_layer_2_w_read_readvariableop(savev2_gcn_layer_3_w_read_readvariableop(savev2_gcn_layer_4_w_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop-savev2_adam_gcn_layer_w_m_read_readvariableop/savev2_adam_gcn_layer_1_w_m_read_readvariableop/savev2_adam_gcn_layer_2_w_m_read_readvariableop/savev2_adam_gcn_layer_3_w_m_read_readvariableop/savev2_adam_gcn_layer_4_w_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop-savev2_adam_gcn_layer_w_v_read_readvariableop/savev2_adam_gcn_layer_1_w_v_read_readvariableop/savev2_adam_gcn_layer_2_w_v_read_readvariableop/savev2_adam_gcn_layer_3_w_v_read_readvariableop/savev2_adam_gcn_layer_4_w_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*Í
_input_shapes»
¸: ::::::::
:
:
:: : : : : : : ::::::::
:
:
:::::::::
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
: 	

_output_shapes
:
:$
 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::$  

_output_shapes

::$! 

_output_shapes

::$" 

_output_shapes

::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:
: &

_output_shapes
:
:$' 

_output_shapes

:
: (

_output_shapes
::)

_output_shapes
: 

ô
C__inference_dense_2_layer_call_and_return_conditional_losses_148387

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
Ü
G__inference_gcn_layer_3_layer_call_and_return_conditional_losses_147565

inputs
inputs_17
%einsum_einsum_readvariableop_resource:
identity

identity_1¢einsum/Einsum/ReadVariableOpy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indicesv
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Sum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xz
truedivRealDivtruediv/x:output:0Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
truediv¢
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02
einsum/Einsum/ReadVariableOp¹
einsum/EinsumEinsum$einsum/Einsum/ReadVariableOp:value:0inputs*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
einsum/Einsum²
einsum/Einsum_1Einsumeinsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
einsum/Einsum_1¶
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
einsum/Einsum_2m
ReluRelueinsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1m
NoOpNoOp^einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã3

C__inference_model_2_layer_call_and_return_conditional_losses_147841

inputs
inputs_1"
gcn_layer_147804:$
gcn_layer_1_147808:$
gcn_layer_2_147812:$
gcn_layer_3_147816:$
gcn_layer_4_147820: 
dense_2_147825:
dense_2_147827: 
dense_3_147830:

dense_3_147832:
 
dense_4_147835:

dense_4_147837:
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢!gcn_layer/StatefulPartitionedCall¢#gcn_layer_1/StatefulPartitionedCall¢#gcn_layer_2/StatefulPartitionedCall¢#gcn_layer_3/StatefulPartitionedCall¢#gcn_layer_4/StatefulPartitionedCallÇ
!gcn_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1gcn_layer_147804*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gcn_layer_layer_call_and_return_conditional_losses_1475022#
!gcn_layer/StatefulPartitionedCall
#gcn_layer_1/StatefulPartitionedCallStatefulPartitionedCall*gcn_layer/StatefulPartitionedCall:output:0*gcn_layer/StatefulPartitionedCall:output:1gcn_layer_1_147808*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_1_layer_call_and_return_conditional_losses_1475232%
#gcn_layer_1/StatefulPartitionedCall
#gcn_layer_2/StatefulPartitionedCallStatefulPartitionedCall,gcn_layer_1/StatefulPartitionedCall:output:0,gcn_layer_1/StatefulPartitionedCall:output:1gcn_layer_2_147812*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_2_layer_call_and_return_conditional_losses_1475442%
#gcn_layer_2/StatefulPartitionedCall
#gcn_layer_3/StatefulPartitionedCallStatefulPartitionedCall,gcn_layer_2/StatefulPartitionedCall:output:0,gcn_layer_2/StatefulPartitionedCall:output:1gcn_layer_3_147816*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_3_layer_call_and_return_conditional_losses_1475652%
#gcn_layer_3/StatefulPartitionedCall
#gcn_layer_4/StatefulPartitionedCallStatefulPartitionedCall,gcn_layer_3/StatefulPartitionedCall:output:0,gcn_layer_3/StatefulPartitionedCall:output:1gcn_layer_4_147820*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_4_layer_call_and_return_conditional_losses_1475862%
#gcn_layer_4/StatefulPartitionedCall¦
GRLayer/PartitionedCallPartitionedCall,gcn_layer_4/StatefulPartitionedCall:output:0,gcn_layer_4/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_GRLayer_layer_call_and_return_conditional_losses_1475982
GRLayer/PartitionedCall©
dense_2/StatefulPartitionedCallStatefulPartitionedCall GRLayer/PartitionedCall:output:0dense_2_147825dense_2_147827*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1476112!
dense_2/StatefulPartitionedCall±
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_147830dense_3_147832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1476282!
dense_3/StatefulPartitionedCall±
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_147835dense_4_147837*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1476452!
dense_4/StatefulPartitionedCall
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityð
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^gcn_layer/StatefulPartitionedCall$^gcn_layer_1/StatefulPartitionedCall$^gcn_layer_2/StatefulPartitionedCall$^gcn_layer_3/StatefulPartitionedCall$^gcn_layer_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!gcn_layer/StatefulPartitionedCall!gcn_layer/StatefulPartitionedCall2J
#gcn_layer_1/StatefulPartitionedCall#gcn_layer_1/StatefulPartitionedCall2J
#gcn_layer_2/StatefulPartitionedCall#gcn_layer_2/StatefulPartitionedCall2J
#gcn_layer_3/StatefulPartitionedCall#gcn_layer_3/StatefulPartitionedCall2J
#gcn_layer_4/StatefulPartitionedCall#gcn_layer_4/StatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å3

C__inference_model_2_layer_call_and_return_conditional_losses_147976
input_2
input_3"
gcn_layer_147939:$
gcn_layer_1_147943:$
gcn_layer_2_147947:$
gcn_layer_3_147951:$
gcn_layer_4_147955: 
dense_2_147960:
dense_2_147962: 
dense_3_147965:

dense_3_147967:
 
dense_4_147970:

dense_4_147972:
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢!gcn_layer/StatefulPartitionedCall¢#gcn_layer_1/StatefulPartitionedCall¢#gcn_layer_2/StatefulPartitionedCall¢#gcn_layer_3/StatefulPartitionedCall¢#gcn_layer_4/StatefulPartitionedCallÇ
!gcn_layer/StatefulPartitionedCallStatefulPartitionedCallinput_2input_3gcn_layer_147939*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gcn_layer_layer_call_and_return_conditional_losses_1475022#
!gcn_layer/StatefulPartitionedCall
#gcn_layer_1/StatefulPartitionedCallStatefulPartitionedCall*gcn_layer/StatefulPartitionedCall:output:0*gcn_layer/StatefulPartitionedCall:output:1gcn_layer_1_147943*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_1_layer_call_and_return_conditional_losses_1475232%
#gcn_layer_1/StatefulPartitionedCall
#gcn_layer_2/StatefulPartitionedCallStatefulPartitionedCall,gcn_layer_1/StatefulPartitionedCall:output:0,gcn_layer_1/StatefulPartitionedCall:output:1gcn_layer_2_147947*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_2_layer_call_and_return_conditional_losses_1475442%
#gcn_layer_2/StatefulPartitionedCall
#gcn_layer_3/StatefulPartitionedCallStatefulPartitionedCall,gcn_layer_2/StatefulPartitionedCall:output:0,gcn_layer_2/StatefulPartitionedCall:output:1gcn_layer_3_147951*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_3_layer_call_and_return_conditional_losses_1475652%
#gcn_layer_3/StatefulPartitionedCall
#gcn_layer_4/StatefulPartitionedCallStatefulPartitionedCall,gcn_layer_3/StatefulPartitionedCall:output:0,gcn_layer_3/StatefulPartitionedCall:output:1gcn_layer_4_147955*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_4_layer_call_and_return_conditional_losses_1475862%
#gcn_layer_4/StatefulPartitionedCall¦
GRLayer/PartitionedCallPartitionedCall,gcn_layer_4/StatefulPartitionedCall:output:0,gcn_layer_4/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_GRLayer_layer_call_and_return_conditional_losses_1475982
GRLayer/PartitionedCall©
dense_2/StatefulPartitionedCallStatefulPartitionedCall GRLayer/PartitionedCall:output:0dense_2_147960dense_2_147962*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1476112!
dense_2/StatefulPartitionedCall±
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_147965dense_3_147967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1476282!
dense_3/StatefulPartitionedCall±
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_147970dense_4_147972*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1476452!
dense_4/StatefulPartitionedCall
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityð
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^gcn_layer/StatefulPartitionedCall$^gcn_layer_1/StatefulPartitionedCall$^gcn_layer_2/StatefulPartitionedCall$^gcn_layer_3/StatefulPartitionedCall$^gcn_layer_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!gcn_layer/StatefulPartitionedCall!gcn_layer/StatefulPartitionedCall2J
#gcn_layer_1/StatefulPartitionedCall#gcn_layer_1/StatefulPartitionedCall2J
#gcn_layer_2/StatefulPartitionedCall#gcn_layer_2/StatefulPartitionedCall2J
#gcn_layer_3/StatefulPartitionedCall#gcn_layer_3/StatefulPartitionedCall2J
#gcn_layer_4/StatefulPartitionedCall#gcn_layer_4/StatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
Ðk
´	
C__inference_model_2_layer_call_and_return_conditional_losses_148146
inputs_0
inputs_1A
/gcn_layer_einsum_einsum_readvariableop_resource:C
1gcn_layer_1_einsum_einsum_readvariableop_resource:C
1gcn_layer_2_einsum_einsum_readvariableop_resource:C
1gcn_layer_3_einsum_einsum_readvariableop_resource:C
1gcn_layer_4_einsum_einsum_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:
5
'dense_3_biasadd_readvariableop_resource:
8
&dense_4_matmul_readvariableop_resource:
5
'dense_4_biasadd_readvariableop_resource:
identity¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢&gcn_layer/einsum/Einsum/ReadVariableOp¢(gcn_layer_1/einsum/Einsum/ReadVariableOp¢(gcn_layer_2/einsum/Einsum/ReadVariableOp¢(gcn_layer_3/einsum/Einsum/ReadVariableOp¢(gcn_layer_4/einsum/Einsum/ReadVariableOp
gcn_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
gcn_layer/Sum/reduction_indices
gcn_layer/SumSuminputs_1(gcn_layer/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer/Sumo
gcn_layer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gcn_layer/truediv/x¢
gcn_layer/truedivRealDivgcn_layer/truediv/x:output:0gcn_layer/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer/truedivÀ
&gcn_layer/einsum/Einsum/ReadVariableOpReadVariableOp/gcn_layer_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02(
&gcn_layer/einsum/Einsum/ReadVariableOpÙ
gcn_layer/einsum/EinsumEinsum.gcn_layer/einsum/Einsum/ReadVariableOp:value:0inputs_0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
gcn_layer/einsum/EinsumÐ
gcn_layer/einsum/Einsum_1Einsum gcn_layer/einsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
gcn_layer/einsum/Einsum_1Þ
gcn_layer/einsum/Einsum_2Einsum"gcn_layer/einsum/Einsum_1:output:0gcn_layer/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
gcn_layer/einsum/Einsum_2
gcn_layer/ReluRelu"gcn_layer/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer/Relu
!gcn_layer_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gcn_layer_1/Sum/reduction_indices
gcn_layer_1/SumSuminputs_1*gcn_layer_1/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_1/Sums
gcn_layer_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gcn_layer_1/truediv/xª
gcn_layer_1/truedivRealDivgcn_layer_1/truediv/x:output:0gcn_layer_1/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_1/truedivÆ
(gcn_layer_1/einsum/Einsum/ReadVariableOpReadVariableOp1gcn_layer_1_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02*
(gcn_layer_1/einsum/Einsum/ReadVariableOpó
gcn_layer_1/einsum/EinsumEinsum0gcn_layer_1/einsum/Einsum/ReadVariableOp:value:0gcn_layer/Relu:activations:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
gcn_layer_1/einsum/EinsumÖ
gcn_layer_1/einsum/Einsum_1Einsum"gcn_layer_1/einsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
gcn_layer_1/einsum/Einsum_1æ
gcn_layer_1/einsum/Einsum_2Einsum$gcn_layer_1/einsum/Einsum_1:output:0gcn_layer_1/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
gcn_layer_1/einsum/Einsum_2
gcn_layer_1/ReluRelu$gcn_layer_1/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_1/Relu
!gcn_layer_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gcn_layer_2/Sum/reduction_indices
gcn_layer_2/SumSuminputs_1*gcn_layer_2/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_2/Sums
gcn_layer_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gcn_layer_2/truediv/xª
gcn_layer_2/truedivRealDivgcn_layer_2/truediv/x:output:0gcn_layer_2/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_2/truedivÆ
(gcn_layer_2/einsum/Einsum/ReadVariableOpReadVariableOp1gcn_layer_2_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02*
(gcn_layer_2/einsum/Einsum/ReadVariableOpõ
gcn_layer_2/einsum/EinsumEinsum0gcn_layer_2/einsum/Einsum/ReadVariableOp:value:0gcn_layer_1/Relu:activations:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
gcn_layer_2/einsum/EinsumÖ
gcn_layer_2/einsum/Einsum_1Einsum"gcn_layer_2/einsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
gcn_layer_2/einsum/Einsum_1æ
gcn_layer_2/einsum/Einsum_2Einsum$gcn_layer_2/einsum/Einsum_1:output:0gcn_layer_2/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
gcn_layer_2/einsum/Einsum_2
gcn_layer_2/ReluRelu$gcn_layer_2/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_2/Relu
!gcn_layer_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gcn_layer_3/Sum/reduction_indices
gcn_layer_3/SumSuminputs_1*gcn_layer_3/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_3/Sums
gcn_layer_3/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gcn_layer_3/truediv/xª
gcn_layer_3/truedivRealDivgcn_layer_3/truediv/x:output:0gcn_layer_3/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_3/truedivÆ
(gcn_layer_3/einsum/Einsum/ReadVariableOpReadVariableOp1gcn_layer_3_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02*
(gcn_layer_3/einsum/Einsum/ReadVariableOpõ
gcn_layer_3/einsum/EinsumEinsum0gcn_layer_3/einsum/Einsum/ReadVariableOp:value:0gcn_layer_2/Relu:activations:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
gcn_layer_3/einsum/EinsumÖ
gcn_layer_3/einsum/Einsum_1Einsum"gcn_layer_3/einsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
gcn_layer_3/einsum/Einsum_1æ
gcn_layer_3/einsum/Einsum_2Einsum$gcn_layer_3/einsum/Einsum_1:output:0gcn_layer_3/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
gcn_layer_3/einsum/Einsum_2
gcn_layer_3/ReluRelu$gcn_layer_3/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_3/Relu
!gcn_layer_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gcn_layer_4/Sum/reduction_indices
gcn_layer_4/SumSuminputs_1*gcn_layer_4/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_4/Sums
gcn_layer_4/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gcn_layer_4/truediv/xª
gcn_layer_4/truedivRealDivgcn_layer_4/truediv/x:output:0gcn_layer_4/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_4/truedivÆ
(gcn_layer_4/einsum/Einsum/ReadVariableOpReadVariableOp1gcn_layer_4_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02*
(gcn_layer_4/einsum/Einsum/ReadVariableOpõ
gcn_layer_4/einsum/EinsumEinsum0gcn_layer_4/einsum/Einsum/ReadVariableOp:value:0gcn_layer_3/Relu:activations:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
gcn_layer_4/einsum/EinsumÖ
gcn_layer_4/einsum/Einsum_1Einsum"gcn_layer_4/einsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
gcn_layer_4/einsum/Einsum_1æ
gcn_layer_4/einsum/Einsum_2Einsum$gcn_layer_4/einsum/Einsum_1:output:0gcn_layer_4/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
gcn_layer_4/einsum/Einsum_2
gcn_layer_4/ReluRelu$gcn_layer_4/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_4/Relu
GRLayer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
GRLayer/Mean/reduction_indices
GRLayer/MeanMeangcn_layer_4/Relu:activations:0'GRLayer/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
GRLayer/Mean¥
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulGRLayer/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu¥
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_3/MatMul¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp¡
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_3/Relu¥
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/MatMul¤
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/BiasAddy
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Softmaxt
IdentityIdentitydense_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp'^gcn_layer/einsum/Einsum/ReadVariableOp)^gcn_layer_1/einsum/Einsum/ReadVariableOp)^gcn_layer_2/einsum/Einsum/ReadVariableOp)^gcn_layer_3/einsum/Einsum/ReadVariableOp)^gcn_layer_4/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2P
&gcn_layer/einsum/Einsum/ReadVariableOp&gcn_layer/einsum/Einsum/ReadVariableOp2T
(gcn_layer_1/einsum/Einsum/ReadVariableOp(gcn_layer_1/einsum/Einsum/ReadVariableOp2T
(gcn_layer_2/einsum/Einsum/ReadVariableOp(gcn_layer_2/einsum/Einsum/ReadVariableOp2T
(gcn_layer_3/einsum/Einsum/ReadVariableOp(gcn_layer_3/einsum/Einsum/ReadVariableOp2T
(gcn_layer_4/einsum/Einsum/ReadVariableOp(gcn_layer_4/einsum/Einsum/ReadVariableOp:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ðk
´	
C__inference_model_2_layer_call_and_return_conditional_losses_148224
inputs_0
inputs_1A
/gcn_layer_einsum_einsum_readvariableop_resource:C
1gcn_layer_1_einsum_einsum_readvariableop_resource:C
1gcn_layer_2_einsum_einsum_readvariableop_resource:C
1gcn_layer_3_einsum_einsum_readvariableop_resource:C
1gcn_layer_4_einsum_einsum_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:
5
'dense_3_biasadd_readvariableop_resource:
8
&dense_4_matmul_readvariableop_resource:
5
'dense_4_biasadd_readvariableop_resource:
identity¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢&gcn_layer/einsum/Einsum/ReadVariableOp¢(gcn_layer_1/einsum/Einsum/ReadVariableOp¢(gcn_layer_2/einsum/Einsum/ReadVariableOp¢(gcn_layer_3/einsum/Einsum/ReadVariableOp¢(gcn_layer_4/einsum/Einsum/ReadVariableOp
gcn_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
gcn_layer/Sum/reduction_indices
gcn_layer/SumSuminputs_1(gcn_layer/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer/Sumo
gcn_layer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gcn_layer/truediv/x¢
gcn_layer/truedivRealDivgcn_layer/truediv/x:output:0gcn_layer/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer/truedivÀ
&gcn_layer/einsum/Einsum/ReadVariableOpReadVariableOp/gcn_layer_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02(
&gcn_layer/einsum/Einsum/ReadVariableOpÙ
gcn_layer/einsum/EinsumEinsum.gcn_layer/einsum/Einsum/ReadVariableOp:value:0inputs_0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
gcn_layer/einsum/EinsumÐ
gcn_layer/einsum/Einsum_1Einsum gcn_layer/einsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
gcn_layer/einsum/Einsum_1Þ
gcn_layer/einsum/Einsum_2Einsum"gcn_layer/einsum/Einsum_1:output:0gcn_layer/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
gcn_layer/einsum/Einsum_2
gcn_layer/ReluRelu"gcn_layer/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer/Relu
!gcn_layer_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gcn_layer_1/Sum/reduction_indices
gcn_layer_1/SumSuminputs_1*gcn_layer_1/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_1/Sums
gcn_layer_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gcn_layer_1/truediv/xª
gcn_layer_1/truedivRealDivgcn_layer_1/truediv/x:output:0gcn_layer_1/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_1/truedivÆ
(gcn_layer_1/einsum/Einsum/ReadVariableOpReadVariableOp1gcn_layer_1_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02*
(gcn_layer_1/einsum/Einsum/ReadVariableOpó
gcn_layer_1/einsum/EinsumEinsum0gcn_layer_1/einsum/Einsum/ReadVariableOp:value:0gcn_layer/Relu:activations:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
gcn_layer_1/einsum/EinsumÖ
gcn_layer_1/einsum/Einsum_1Einsum"gcn_layer_1/einsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
gcn_layer_1/einsum/Einsum_1æ
gcn_layer_1/einsum/Einsum_2Einsum$gcn_layer_1/einsum/Einsum_1:output:0gcn_layer_1/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
gcn_layer_1/einsum/Einsum_2
gcn_layer_1/ReluRelu$gcn_layer_1/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_1/Relu
!gcn_layer_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gcn_layer_2/Sum/reduction_indices
gcn_layer_2/SumSuminputs_1*gcn_layer_2/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_2/Sums
gcn_layer_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gcn_layer_2/truediv/xª
gcn_layer_2/truedivRealDivgcn_layer_2/truediv/x:output:0gcn_layer_2/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_2/truedivÆ
(gcn_layer_2/einsum/Einsum/ReadVariableOpReadVariableOp1gcn_layer_2_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02*
(gcn_layer_2/einsum/Einsum/ReadVariableOpõ
gcn_layer_2/einsum/EinsumEinsum0gcn_layer_2/einsum/Einsum/ReadVariableOp:value:0gcn_layer_1/Relu:activations:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
gcn_layer_2/einsum/EinsumÖ
gcn_layer_2/einsum/Einsum_1Einsum"gcn_layer_2/einsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
gcn_layer_2/einsum/Einsum_1æ
gcn_layer_2/einsum/Einsum_2Einsum$gcn_layer_2/einsum/Einsum_1:output:0gcn_layer_2/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
gcn_layer_2/einsum/Einsum_2
gcn_layer_2/ReluRelu$gcn_layer_2/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_2/Relu
!gcn_layer_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gcn_layer_3/Sum/reduction_indices
gcn_layer_3/SumSuminputs_1*gcn_layer_3/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_3/Sums
gcn_layer_3/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gcn_layer_3/truediv/xª
gcn_layer_3/truedivRealDivgcn_layer_3/truediv/x:output:0gcn_layer_3/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_3/truedivÆ
(gcn_layer_3/einsum/Einsum/ReadVariableOpReadVariableOp1gcn_layer_3_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02*
(gcn_layer_3/einsum/Einsum/ReadVariableOpõ
gcn_layer_3/einsum/EinsumEinsum0gcn_layer_3/einsum/Einsum/ReadVariableOp:value:0gcn_layer_2/Relu:activations:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
gcn_layer_3/einsum/EinsumÖ
gcn_layer_3/einsum/Einsum_1Einsum"gcn_layer_3/einsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
gcn_layer_3/einsum/Einsum_1æ
gcn_layer_3/einsum/Einsum_2Einsum$gcn_layer_3/einsum/Einsum_1:output:0gcn_layer_3/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
gcn_layer_3/einsum/Einsum_2
gcn_layer_3/ReluRelu$gcn_layer_3/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_3/Relu
!gcn_layer_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gcn_layer_4/Sum/reduction_indices
gcn_layer_4/SumSuminputs_1*gcn_layer_4/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_4/Sums
gcn_layer_4/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gcn_layer_4/truediv/xª
gcn_layer_4/truedivRealDivgcn_layer_4/truediv/x:output:0gcn_layer_4/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_4/truedivÆ
(gcn_layer_4/einsum/Einsum/ReadVariableOpReadVariableOp1gcn_layer_4_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02*
(gcn_layer_4/einsum/Einsum/ReadVariableOpõ
gcn_layer_4/einsum/EinsumEinsum0gcn_layer_4/einsum/Einsum/ReadVariableOp:value:0gcn_layer_3/Relu:activations:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
gcn_layer_4/einsum/EinsumÖ
gcn_layer_4/einsum/Einsum_1Einsum"gcn_layer_4/einsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
gcn_layer_4/einsum/Einsum_1æ
gcn_layer_4/einsum/Einsum_2Einsum$gcn_layer_4/einsum/Einsum_1:output:0gcn_layer_4/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
gcn_layer_4/einsum/Einsum_2
gcn_layer_4/ReluRelu$gcn_layer_4/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gcn_layer_4/Relu
GRLayer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
GRLayer/Mean/reduction_indices
GRLayer/MeanMeangcn_layer_4/Relu:activations:0'GRLayer/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
GRLayer/Mean¥
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulGRLayer/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu¥
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_3/MatMul¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_3/BiasAdd/ReadVariableOp¡
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_3/Relu¥
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/MatMul¤
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/BiasAddy
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Softmaxt
IdentityIdentitydense_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp'^gcn_layer/einsum/Einsum/ReadVariableOp)^gcn_layer_1/einsum/Einsum/ReadVariableOp)^gcn_layer_2/einsum/Einsum/ReadVariableOp)^gcn_layer_3/einsum/Einsum/ReadVariableOp)^gcn_layer_4/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2P
&gcn_layer/einsum/Einsum/ReadVariableOp&gcn_layer/einsum/Einsum/ReadVariableOp2T
(gcn_layer_1/einsum/Einsum/ReadVariableOp(gcn_layer_1/einsum/Einsum/ReadVariableOp2T
(gcn_layer_2/einsum/Einsum/ReadVariableOp(gcn_layer_2/einsum/Einsum/ReadVariableOp2T
(gcn_layer_3/einsum/Einsum/ReadVariableOp(gcn_layer_3/einsum/Einsum/ReadVariableOp2T
(gcn_layer_4/einsum/Einsum/ReadVariableOp(gcn_layer_4/einsum/Einsum/ReadVariableOp:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
{
À

!__inference__wrapped_model_147477
input_2
input_3I
7model_2_gcn_layer_einsum_einsum_readvariableop_resource:K
9model_2_gcn_layer_1_einsum_einsum_readvariableop_resource:K
9model_2_gcn_layer_2_einsum_einsum_readvariableop_resource:K
9model_2_gcn_layer_3_einsum_einsum_readvariableop_resource:K
9model_2_gcn_layer_4_einsum_einsum_readvariableop_resource:@
.model_2_dense_2_matmul_readvariableop_resource:=
/model_2_dense_2_biasadd_readvariableop_resource:@
.model_2_dense_3_matmul_readvariableop_resource:
=
/model_2_dense_3_biasadd_readvariableop_resource:
@
.model_2_dense_4_matmul_readvariableop_resource:
=
/model_2_dense_4_biasadd_readvariableop_resource:
identity¢&model_2/dense_2/BiasAdd/ReadVariableOp¢%model_2/dense_2/MatMul/ReadVariableOp¢&model_2/dense_3/BiasAdd/ReadVariableOp¢%model_2/dense_3/MatMul/ReadVariableOp¢&model_2/dense_4/BiasAdd/ReadVariableOp¢%model_2/dense_4/MatMul/ReadVariableOp¢.model_2/gcn_layer/einsum/Einsum/ReadVariableOp¢0model_2/gcn_layer_1/einsum/Einsum/ReadVariableOp¢0model_2/gcn_layer_2/einsum/Einsum/ReadVariableOp¢0model_2/gcn_layer_3/einsum/Einsum/ReadVariableOp¢0model_2/gcn_layer_4/einsum/Einsum/ReadVariableOp
'model_2/gcn_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_2/gcn_layer/Sum/reduction_indices«
model_2/gcn_layer/SumSuminput_30model_2/gcn_layer/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer/Sum
model_2/gcn_layer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
model_2/gcn_layer/truediv/xÂ
model_2/gcn_layer/truedivRealDiv$model_2/gcn_layer/truediv/x:output:0model_2/gcn_layer/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer/truedivØ
.model_2/gcn_layer/einsum/Einsum/ReadVariableOpReadVariableOp7model_2_gcn_layer_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype020
.model_2/gcn_layer/einsum/Einsum/ReadVariableOpð
model_2/gcn_layer/einsum/EinsumEinsum6model_2/gcn_layer/einsum/Einsum/ReadVariableOp:value:0input_2*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2!
model_2/gcn_layer/einsum/Einsumç
!model_2/gcn_layer/einsum/Einsum_1Einsum(model_2/gcn_layer/einsum/Einsum:output:0input_3*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2#
!model_2/gcn_layer/einsum/Einsum_1þ
!model_2/gcn_layer/einsum/Einsum_2Einsum*model_2/gcn_layer/einsum/Einsum_1:output:0model_2/gcn_layer/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2#
!model_2/gcn_layer/einsum/Einsum_2£
model_2/gcn_layer/ReluRelu*model_2/gcn_layer/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer/Relu¡
)model_2/gcn_layer_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)model_2/gcn_layer_1/Sum/reduction_indices±
model_2/gcn_layer_1/SumSuminput_32model_2/gcn_layer_1/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer_1/Sum
model_2/gcn_layer_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
model_2/gcn_layer_1/truediv/xÊ
model_2/gcn_layer_1/truedivRealDiv&model_2/gcn_layer_1/truediv/x:output:0 model_2/gcn_layer_1/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer_1/truedivÞ
0model_2/gcn_layer_1/einsum/Einsum/ReadVariableOpReadVariableOp9model_2_gcn_layer_1_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype022
0model_2/gcn_layer_1/einsum/Einsum/ReadVariableOp
!model_2/gcn_layer_1/einsum/EinsumEinsum8model_2/gcn_layer_1/einsum/Einsum/ReadVariableOp:value:0$model_2/gcn_layer/Relu:activations:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2#
!model_2/gcn_layer_1/einsum/Einsumí
#model_2/gcn_layer_1/einsum/Einsum_1Einsum*model_2/gcn_layer_1/einsum/Einsum:output:0input_3*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2%
#model_2/gcn_layer_1/einsum/Einsum_1
#model_2/gcn_layer_1/einsum/Einsum_2Einsum,model_2/gcn_layer_1/einsum/Einsum_1:output:0model_2/gcn_layer_1/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2%
#model_2/gcn_layer_1/einsum/Einsum_2©
model_2/gcn_layer_1/ReluRelu,model_2/gcn_layer_1/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer_1/Relu¡
)model_2/gcn_layer_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)model_2/gcn_layer_2/Sum/reduction_indices±
model_2/gcn_layer_2/SumSuminput_32model_2/gcn_layer_2/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer_2/Sum
model_2/gcn_layer_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
model_2/gcn_layer_2/truediv/xÊ
model_2/gcn_layer_2/truedivRealDiv&model_2/gcn_layer_2/truediv/x:output:0 model_2/gcn_layer_2/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer_2/truedivÞ
0model_2/gcn_layer_2/einsum/Einsum/ReadVariableOpReadVariableOp9model_2_gcn_layer_2_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype022
0model_2/gcn_layer_2/einsum/Einsum/ReadVariableOp
!model_2/gcn_layer_2/einsum/EinsumEinsum8model_2/gcn_layer_2/einsum/Einsum/ReadVariableOp:value:0&model_2/gcn_layer_1/Relu:activations:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2#
!model_2/gcn_layer_2/einsum/Einsumí
#model_2/gcn_layer_2/einsum/Einsum_1Einsum*model_2/gcn_layer_2/einsum/Einsum:output:0input_3*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2%
#model_2/gcn_layer_2/einsum/Einsum_1
#model_2/gcn_layer_2/einsum/Einsum_2Einsum,model_2/gcn_layer_2/einsum/Einsum_1:output:0model_2/gcn_layer_2/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2%
#model_2/gcn_layer_2/einsum/Einsum_2©
model_2/gcn_layer_2/ReluRelu,model_2/gcn_layer_2/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer_2/Relu¡
)model_2/gcn_layer_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)model_2/gcn_layer_3/Sum/reduction_indices±
model_2/gcn_layer_3/SumSuminput_32model_2/gcn_layer_3/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer_3/Sum
model_2/gcn_layer_3/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
model_2/gcn_layer_3/truediv/xÊ
model_2/gcn_layer_3/truedivRealDiv&model_2/gcn_layer_3/truediv/x:output:0 model_2/gcn_layer_3/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer_3/truedivÞ
0model_2/gcn_layer_3/einsum/Einsum/ReadVariableOpReadVariableOp9model_2_gcn_layer_3_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype022
0model_2/gcn_layer_3/einsum/Einsum/ReadVariableOp
!model_2/gcn_layer_3/einsum/EinsumEinsum8model_2/gcn_layer_3/einsum/Einsum/ReadVariableOp:value:0&model_2/gcn_layer_2/Relu:activations:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2#
!model_2/gcn_layer_3/einsum/Einsumí
#model_2/gcn_layer_3/einsum/Einsum_1Einsum*model_2/gcn_layer_3/einsum/Einsum:output:0input_3*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2%
#model_2/gcn_layer_3/einsum/Einsum_1
#model_2/gcn_layer_3/einsum/Einsum_2Einsum,model_2/gcn_layer_3/einsum/Einsum_1:output:0model_2/gcn_layer_3/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2%
#model_2/gcn_layer_3/einsum/Einsum_2©
model_2/gcn_layer_3/ReluRelu,model_2/gcn_layer_3/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer_3/Relu¡
)model_2/gcn_layer_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)model_2/gcn_layer_4/Sum/reduction_indices±
model_2/gcn_layer_4/SumSuminput_32model_2/gcn_layer_4/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer_4/Sum
model_2/gcn_layer_4/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
model_2/gcn_layer_4/truediv/xÊ
model_2/gcn_layer_4/truedivRealDiv&model_2/gcn_layer_4/truediv/x:output:0 model_2/gcn_layer_4/Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer_4/truedivÞ
0model_2/gcn_layer_4/einsum/Einsum/ReadVariableOpReadVariableOp9model_2_gcn_layer_4_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype022
0model_2/gcn_layer_4/einsum/Einsum/ReadVariableOp
!model_2/gcn_layer_4/einsum/EinsumEinsum8model_2/gcn_layer_4/einsum/Einsum/ReadVariableOp:value:0&model_2/gcn_layer_3/Relu:activations:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2#
!model_2/gcn_layer_4/einsum/Einsumí
#model_2/gcn_layer_4/einsum/Einsum_1Einsum*model_2/gcn_layer_4/einsum/Einsum:output:0input_3*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2%
#model_2/gcn_layer_4/einsum/Einsum_1
#model_2/gcn_layer_4/einsum/Einsum_2Einsum,model_2/gcn_layer_4/einsum/Einsum_1:output:0model_2/gcn_layer_4/truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2%
#model_2/gcn_layer_4/einsum/Einsum_2©
model_2/gcn_layer_4/ReluRelu,model_2/gcn_layer_4/einsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/gcn_layer_4/Relu
&model_2/GRLayer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_2/GRLayer/Mean/reduction_indices¿
model_2/GRLayer/MeanMean&model_2/gcn_layer_4/Relu:activations:0/model_2/GRLayer/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/GRLayer/Mean½
%model_2/dense_2/MatMul/ReadVariableOpReadVariableOp.model_2_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%model_2/dense_2/MatMul/ReadVariableOpº
model_2/dense_2/MatMulMatMulmodel_2/GRLayer/Mean:output:0-model_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/dense_2/MatMul¼
&model_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_2/dense_2/BiasAdd/ReadVariableOpÁ
model_2/dense_2/BiasAddBiasAdd model_2/dense_2/MatMul:product:0.model_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/dense_2/BiasAdd
model_2/dense_2/ReluRelu model_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/dense_2/Relu½
%model_2/dense_3/MatMul/ReadVariableOpReadVariableOp.model_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02'
%model_2/dense_3/MatMul/ReadVariableOp¿
model_2/dense_3/MatMulMatMul"model_2/dense_2/Relu:activations:0-model_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
model_2/dense_3/MatMul¼
&model_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02(
&model_2/dense_3/BiasAdd/ReadVariableOpÁ
model_2/dense_3/BiasAddBiasAdd model_2/dense_3/MatMul:product:0.model_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
model_2/dense_3/BiasAdd
model_2/dense_3/ReluRelu model_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
model_2/dense_3/Relu½
%model_2/dense_4/MatMul/ReadVariableOpReadVariableOp.model_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02'
%model_2/dense_4/MatMul/ReadVariableOp¿
model_2/dense_4/MatMulMatMul"model_2/dense_3/Relu:activations:0-model_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/dense_4/MatMul¼
&model_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_2/dense_4/BiasAdd/ReadVariableOpÁ
model_2/dense_4/BiasAddBiasAdd model_2/dense_4/MatMul:product:0.model_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/dense_4/BiasAdd
model_2/dense_4/SoftmaxSoftmax model_2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/dense_4/Softmax|
IdentityIdentity!model_2/dense_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¾
NoOpNoOp'^model_2/dense_2/BiasAdd/ReadVariableOp&^model_2/dense_2/MatMul/ReadVariableOp'^model_2/dense_3/BiasAdd/ReadVariableOp&^model_2/dense_3/MatMul/ReadVariableOp'^model_2/dense_4/BiasAdd/ReadVariableOp&^model_2/dense_4/MatMul/ReadVariableOp/^model_2/gcn_layer/einsum/Einsum/ReadVariableOp1^model_2/gcn_layer_1/einsum/Einsum/ReadVariableOp1^model_2/gcn_layer_2/einsum/Einsum/ReadVariableOp1^model_2/gcn_layer_3/einsum/Einsum/ReadVariableOp1^model_2/gcn_layer_4/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2P
&model_2/dense_2/BiasAdd/ReadVariableOp&model_2/dense_2/BiasAdd/ReadVariableOp2N
%model_2/dense_2/MatMul/ReadVariableOp%model_2/dense_2/MatMul/ReadVariableOp2P
&model_2/dense_3/BiasAdd/ReadVariableOp&model_2/dense_3/BiasAdd/ReadVariableOp2N
%model_2/dense_3/MatMul/ReadVariableOp%model_2/dense_3/MatMul/ReadVariableOp2P
&model_2/dense_4/BiasAdd/ReadVariableOp&model_2/dense_4/BiasAdd/ReadVariableOp2N
%model_2/dense_4/MatMul/ReadVariableOp%model_2/dense_4/MatMul/ReadVariableOp2`
.model_2/gcn_layer/einsum/Einsum/ReadVariableOp.model_2/gcn_layer/einsum/Einsum/ReadVariableOp2d
0model_2/gcn_layer_1/einsum/Einsum/ReadVariableOp0model_2/gcn_layer_1/einsum/Einsum/ReadVariableOp2d
0model_2/gcn_layer_2/einsum/Einsum/ReadVariableOp0model_2/gcn_layer_2/einsum/Einsum/ReadVariableOp2d
0model_2/gcn_layer_3/einsum/Einsum/ReadVariableOp0model_2/gcn_layer_3/einsum/Einsum/ReadVariableOp2d
0model_2/gcn_layer_4/einsum/Einsum/ReadVariableOp0model_2/gcn_layer_4/einsum/Einsum/ReadVariableOp:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
¢
Ü
G__inference_gcn_layer_1_layer_call_and_return_conditional_losses_147523

inputs
inputs_17
%einsum_einsum_readvariableop_resource:
identity

identity_1¢einsum/Einsum/ReadVariableOpy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indicesv
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Sum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xz
truedivRealDivtruediv/x:output:0Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
truediv¢
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02
einsum/Einsum/ReadVariableOp¹
einsum/EinsumEinsum$einsum/Einsum/ReadVariableOp:value:0inputs*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
einsum/Einsum²
einsum/Einsum_1Einsumeinsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
einsum/Einsum_1¶
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
einsum/Einsum_2m
ReluRelueinsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1m
NoOpNoOp^einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
o
C__inference_GRLayer_layer_call_and_return_conditional_losses_148367
inputs_0
inputs_1
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesq
MeanMeaninputs_0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1

ô
C__inference_dense_4_layer_call_and_return_conditional_losses_147645

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Å3

C__inference_model_2_layer_call_and_return_conditional_losses_147935
input_2
input_3"
gcn_layer_147898:$
gcn_layer_1_147902:$
gcn_layer_2_147906:$
gcn_layer_3_147910:$
gcn_layer_4_147914: 
dense_2_147919:
dense_2_147921: 
dense_3_147924:

dense_3_147926:
 
dense_4_147929:

dense_4_147931:
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢!gcn_layer/StatefulPartitionedCall¢#gcn_layer_1/StatefulPartitionedCall¢#gcn_layer_2/StatefulPartitionedCall¢#gcn_layer_3/StatefulPartitionedCall¢#gcn_layer_4/StatefulPartitionedCallÇ
!gcn_layer/StatefulPartitionedCallStatefulPartitionedCallinput_2input_3gcn_layer_147898*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gcn_layer_layer_call_and_return_conditional_losses_1475022#
!gcn_layer/StatefulPartitionedCall
#gcn_layer_1/StatefulPartitionedCallStatefulPartitionedCall*gcn_layer/StatefulPartitionedCall:output:0*gcn_layer/StatefulPartitionedCall:output:1gcn_layer_1_147902*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_1_layer_call_and_return_conditional_losses_1475232%
#gcn_layer_1/StatefulPartitionedCall
#gcn_layer_2/StatefulPartitionedCallStatefulPartitionedCall,gcn_layer_1/StatefulPartitionedCall:output:0,gcn_layer_1/StatefulPartitionedCall:output:1gcn_layer_2_147906*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_2_layer_call_and_return_conditional_losses_1475442%
#gcn_layer_2/StatefulPartitionedCall
#gcn_layer_3/StatefulPartitionedCallStatefulPartitionedCall,gcn_layer_2/StatefulPartitionedCall:output:0,gcn_layer_2/StatefulPartitionedCall:output:1gcn_layer_3_147910*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_3_layer_call_and_return_conditional_losses_1475652%
#gcn_layer_3/StatefulPartitionedCall
#gcn_layer_4/StatefulPartitionedCallStatefulPartitionedCall,gcn_layer_3/StatefulPartitionedCall:output:0,gcn_layer_3/StatefulPartitionedCall:output:1gcn_layer_4_147914*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gcn_layer_4_layer_call_and_return_conditional_losses_1475862%
#gcn_layer_4/StatefulPartitionedCall¦
GRLayer/PartitionedCallPartitionedCall,gcn_layer_4/StatefulPartitionedCall:output:0,gcn_layer_4/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_GRLayer_layer_call_and_return_conditional_losses_1475982
GRLayer/PartitionedCall©
dense_2/StatefulPartitionedCallStatefulPartitionedCall GRLayer/PartitionedCall:output:0dense_2_147919dense_2_147921*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1476112!
dense_2/StatefulPartitionedCall±
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_147924dense_3_147926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1476282!
dense_3/StatefulPartitionedCall±
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_147929dense_4_147931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1476452!
dense_4/StatefulPartitionedCall
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityð
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^gcn_layer/StatefulPartitionedCall$^gcn_layer_1/StatefulPartitionedCall$^gcn_layer_2/StatefulPartitionedCall$^gcn_layer_3/StatefulPartitionedCall$^gcn_layer_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!gcn_layer/StatefulPartitionedCall!gcn_layer/StatefulPartitionedCall2J
#gcn_layer_1/StatefulPartitionedCall#gcn_layer_1/StatefulPartitionedCall2J
#gcn_layer_2/StatefulPartitionedCall#gcn_layer_2/StatefulPartitionedCall2J
#gcn_layer_3/StatefulPartitionedCall#gcn_layer_3/StatefulPartitionedCall2J
#gcn_layer_4/StatefulPartitionedCall#gcn_layer_4/StatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
ª
Þ
G__inference_gcn_layer_2_layer_call_and_return_conditional_losses_148302
inputs_0
inputs_17
%einsum_einsum_readvariableop_resource:
identity

identity_1¢einsum/Einsum/ReadVariableOpy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indicesv
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Sum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	truediv/xz
truedivRealDivtruediv/x:output:0Sum:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
truediv¢
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02
einsum/Einsum/ReadVariableOp»
einsum/EinsumEinsum$einsum/Einsum/ReadVariableOp:value:0inputs_0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationkl,bjk->lbj2
einsum/Einsum²
einsum/Einsum_1Einsumeinsum/Einsum:output:0inputs_1*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbj,bij->lbi2
einsum/Einsum_1¶
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0truediv:z:0*
N*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
equationlbi,bi->bil2
einsum/Einsum_2m
ReluRelueinsum/Einsum_2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1m
NoOpNoOp^einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultö
H
input_2=
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Q
input_3F
serving_default_input_3:0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ;
dense_40
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¦°
Æ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
__call__
+&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
®
w
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
®
w
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
®
w
regularization_losses
trainable_variables
	variables
 	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
®
!w
"regularization_losses
#trainable_variables
$	variables
%	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
®
&w
'regularization_losses
(trainable_variables
)	variables
*	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
+regularization_losses
,trainable_variables
-	variables
.	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"
_tf_keras_layer
½

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"
_tf_keras_layer
½

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"
_tf_keras_layer
½

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer
º
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratem}m~m!m&m/m0m5m6m;m<mvvv!v&v/v0v5v6v;v<v"
tf_deprecated_optimizer
 "
trackable_list_wrapper
n
0
1
2
!3
&4
/5
06
57
68
;9
<10"
trackable_list_wrapper
n
0
1
2
!3
&4
/5
06
57
68
;9
<10"
trackable_list_wrapper
Î
Fnon_trainable_variables
regularization_losses
trainable_variables
Glayer_regularization_losses
Hlayer_metrics
	variables
Imetrics

Jlayers
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
¨serving_default"
signature_map
:2gcn_layer/w
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
Knon_trainable_variables
regularization_losses
Llayer_metrics
Mlayer_regularization_losses
trainable_variables
	variables
Nmetrics

Olayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:2gcn_layer_1/w
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
Pnon_trainable_variables
regularization_losses
Qlayer_metrics
Rlayer_regularization_losses
trainable_variables
	variables
Smetrics

Tlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:2gcn_layer_2/w
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
Unon_trainable_variables
regularization_losses
Vlayer_metrics
Wlayer_regularization_losses
trainable_variables
	variables
Xmetrics

Ylayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:2gcn_layer_3/w
 "
trackable_list_wrapper
'
!0"
trackable_list_wrapper
'
!0"
trackable_list_wrapper
°
Znon_trainable_variables
"regularization_losses
[layer_metrics
\layer_regularization_losses
#trainable_variables
$	variables
]metrics

^layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:2gcn_layer_4/w
 "
trackable_list_wrapper
'
&0"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
°
_non_trainable_variables
'regularization_losses
`layer_metrics
alayer_regularization_losses
(trainable_variables
)	variables
bmetrics

clayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
dnon_trainable_variables
+regularization_losses
elayer_metrics
flayer_regularization_losses
,trainable_variables
-	variables
gmetrics

hlayers
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
°
inon_trainable_variables
1regularization_losses
jlayer_metrics
klayer_regularization_losses
2trainable_variables
3	variables
lmetrics

mlayers
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 :
2dense_3/kernel
:
2dense_3/bias
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
°
nnon_trainable_variables
7regularization_losses
olayer_metrics
player_regularization_losses
8trainable_variables
9	variables
qmetrics

rlayers
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
 :
2dense_4/kernel
:2dense_4/bias
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
°
snon_trainable_variables
=regularization_losses
tlayer_metrics
ulayer_regularization_losses
>trainable_variables
?	variables
vmetrics

wlayers
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
x0"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
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
N
	ytotal
	zcount
{	variables
|	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
y0
z1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
": 2Adam/gcn_layer/w/m
$:"2Adam/gcn_layer_1/w/m
$:"2Adam/gcn_layer_2/w/m
$:"2Adam/gcn_layer_3/w/m
$:"2Adam/gcn_layer_4/w/m
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
%:#
2Adam/dense_3/kernel/m
:
2Adam/dense_3/bias/m
%:#
2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
": 2Adam/gcn_layer/w/v
$:"2Adam/gcn_layer_1/w/v
$:"2Adam/gcn_layer_2/w/v
$:"2Adam/gcn_layer_3/w/v
$:"2Adam/gcn_layer_4/w/v
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
%:#
2Adam/dense_3/kernel/v
:
2Adam/dense_3/bias/v
%:#
2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
î2ë
(__inference_model_2_layer_call_fn_147677
(__inference_model_2_layer_call_fn_148040
(__inference_model_2_layer_call_fn_148068
(__inference_model_2_layer_call_fn_147894À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
C__inference_model_2_layer_call_and_return_conditional_losses_148146
C__inference_model_2_layer_call_and_return_conditional_losses_148224
C__inference_model_2_layer_call_and_return_conditional_losses_147935
C__inference_model_2_layer_call_and_return_conditional_losses_147976À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ª2§
!__inference__wrapped_model_147477
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *q¢n
l¢i
.+
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
74
input_3'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô2Ñ
*__inference_gcn_layer_layer_call_fn_148234¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_gcn_layer_layer_call_and_return_conditional_losses_148250¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_gcn_layer_1_layer_call_fn_148260¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_gcn_layer_1_layer_call_and_return_conditional_losses_148276¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_gcn_layer_2_layer_call_fn_148286¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_gcn_layer_2_layer_call_and_return_conditional_losses_148302¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_gcn_layer_3_layer_call_fn_148312¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_gcn_layer_3_layer_call_and_return_conditional_losses_148328¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_gcn_layer_4_layer_call_fn_148338¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_gcn_layer_4_layer_call_and_return_conditional_losses_148354¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_GRLayer_layer_call_fn_148360¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_GRLayer_layer_call_and_return_conditional_losses_148367¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_2_layer_call_fn_148376¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_2_layer_call_and_return_conditional_losses_148387¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_3_layer_call_fn_148396¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_3_layer_call_and_return_conditional_losses_148407¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_4_layer_call_fn_148416¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_4_layer_call_and_return_conditional_losses_148427¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÒBÏ
$__inference_signature_wrapper_148012input_2input_3"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 î
C__inference_GRLayer_layer_call_and_return_conditional_losses_148367¦}¢z
s¢p
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Æ
(__inference_GRLayer_layer_call_fn_148360}¢z
s¢p
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿã
!__inference__wrapped_model_147477½!&/056;<{¢x
q¢n
l¢i
.+
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
74
input_3'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_4!
dense_4ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_2_layer_call_and_return_conditional_losses_148387\/0/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_2_layer_call_fn_148376O/0/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_3_layer_call_and_return_conditional_losses_148407\56/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 {
(__inference_dense_3_layer_call_fn_148396O56/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
£
C__inference_dense_4_layer_call_and_return_conditional_losses_148427\;</¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_4_layer_call_fn_148416O;</¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ¾
G__inference_gcn_layer_1_layer_call_and_return_conditional_losses_148276ò}¢z
s¢p
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "n¢k
d¢a
*'
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
30
0/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
,__inference_gcn_layer_1_layer_call_fn_148260ä}¢z
s¢p
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "`¢]
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1.
1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
G__inference_gcn_layer_2_layer_call_and_return_conditional_losses_148302ò}¢z
s¢p
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "n¢k
d¢a
*'
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
30
0/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
,__inference_gcn_layer_2_layer_call_fn_148286ä}¢z
s¢p
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "`¢]
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1.
1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
G__inference_gcn_layer_3_layer_call_and_return_conditional_losses_148328ò!}¢z
s¢p
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "n¢k
d¢a
*'
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
30
0/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
,__inference_gcn_layer_3_layer_call_fn_148312ä!}¢z
s¢p
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "`¢]
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1.
1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
G__inference_gcn_layer_4_layer_call_and_return_conditional_losses_148354ò&}¢z
s¢p
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "n¢k
d¢a
*'
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
30
0/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
,__inference_gcn_layer_4_layer_call_fn_148338ä&}¢z
s¢p
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "`¢]
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1.
1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
E__inference_gcn_layer_layer_call_and_return_conditional_losses_148250ò}¢z
s¢p
nk
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "n¢k
d¢a
*'
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
30
0/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
*__inference_gcn_layer_layer_call_fn_148234ä}¢z
s¢p
nk
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "`¢]
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1.
1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
C__inference_model_2_layer_call_and_return_conditional_losses_147935»!&/056;<¢
y¢v
l¢i
.+
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
74
input_3'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_2_layer_call_and_return_conditional_losses_147976»!&/056;<¢
y¢v
l¢i
.+
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
74
input_3'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_2_layer_call_and_return_conditional_losses_148146½!&/056;<¢
{¢x
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_2_layer_call_and_return_conditional_losses_148224½!&/056;<¢
{¢x
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Û
(__inference_model_2_layer_call_fn_147677®!&/056;<¢
y¢v
l¢i
.+
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
74
input_3'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÛ
(__inference_model_2_layer_call_fn_147894®!&/056;<¢
y¢v
l¢i
.+
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
74
input_3'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÝ
(__inference_model_2_layer_call_fn_148040°!&/056;<¢
{¢x
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÝ
(__inference_model_2_layer_call_fn_148068°!&/056;<¢
{¢x
n¢k
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
inputs/1'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿú
$__inference_signature_wrapper_148012Ñ!&/056;<¢
¢ 
ª
9
input_2.+
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
B
input_374
input_3'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_4!
dense_4ÿÿÿÿÿÿÿÿÿ