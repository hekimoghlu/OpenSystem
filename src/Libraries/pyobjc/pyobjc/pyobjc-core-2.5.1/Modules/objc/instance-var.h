/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

#ifndef OBJC_INSTANCE_VAR
#define OBJC_INSTANCE_VAR

typedef struct {
	PyObject_HEAD
	char* name;      /* Name of the instance variable */
	char* type;      /* Type of the instance variable for definition only */
	int   isOutlet;
	int   isSlot;
	Ivar   ivar;
} PyObjCInstanceVariable;


extern PyTypeObject PyObjCInstanceVariable_Type;
#define PyObjCInstanceVariable_Check(obj) PyObject_TypeCheck((obj), &PyObjCInstanceVariable_Type)

PyObject* PyObjCInstanceVariable_New(char* name);
int	  PyObjCInstanceVariable_SetName(PyObject* self, PyObject* name);

#define PyObjCInstanceVariable_IsOutlet(obj) \
	(((PyObjCInstanceVariable*)(obj))->isOutlet)
#define PyObjCInstanceVariable_IsSlot(obj) \
	(((PyObjCInstanceVariable*)(obj))->isSlot)
#define PyObjCInstanceVariable_GetName(obj) \
	(((PyObjCInstanceVariable*)(obj))->name)
#define PyObjCInstanceVariable_GetType(obj) \
	(((PyObjCInstanceVariable*)(obj))->type)

PyObject* PyObjCIvar_Info(PyObject* self, PyObject* arg);
PyObject* PyObjCIvar_Set(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* PyObjCIvar_Get(PyObject* self, PyObject* args, PyObject* kwds);



#endif /* OBJC_INSTANCE_VAR */
