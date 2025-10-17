/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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
#if PY_MAJOR_VERSION == 2 && defined(USE_TOOLBOX_OBJECT_GLUE)

#ifndef __LP64__

#include "pymactoolbox.h"

#else
	/* FIXME: the bits of pymactoolbox.h that we need,
	 * because said header doesn't work in 64-bit mode
	 */
extern PyObject *WinObj_New(WindowPtr);
extern int WinObj_Convert(PyObject *, WindowPtr *);
extern PyObject *WinObj_WhichWindow(WindowPtr);

#endif

static int
py2window(PyObject* obj, void* output)
{
	return WinObj_Convert(obj, (WindowPtr*)output);
}

static PyObject*
window2py(void* value)
{
	return WinObj_New((WindowPtr)value);
}

#endif /* PY_MAJOR_VERSION == 2 */

static int setup_carbon(PyObject* m __attribute__((__unused__)))
{
#if PY_MAJOR_VERSION == 2 && defined(USE_TOOLBOX_OBJECT_GLUE)
	if (PyObjCPointerWrapper_Register(@encode(WindowRef),
	                &window2py, &py2window) < 0)
		return -1;
#endif

	return 0;
}
