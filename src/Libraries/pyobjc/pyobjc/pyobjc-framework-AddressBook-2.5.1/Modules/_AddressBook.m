/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 18, 2023.
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

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "pyobjc-api.h"

#import <AddressBook/AddressBook.h>

/* We include the source code here instead of 
 * using the linker due to limitations in pyobjc-api.h
 */
#include "_AddressBook_protocols.m"


static PyMethodDef mod_methods[] = {
	{ 0, 0, 0, 0 } /* sentinel */
};


/* Python glue */
PyObjC_MODULE_INIT(_AddressBook)
{
	PyObject* m;
	m = PyObjC_MODULE_CREATE(_AddressBook)
	if (!m) { 
		PyObjC_INITERROR();
	}

	if (PyObjC_ImportAPI(m) == -1) PyObjC_INITERROR();

	PyObjC_INITDONE();
}
