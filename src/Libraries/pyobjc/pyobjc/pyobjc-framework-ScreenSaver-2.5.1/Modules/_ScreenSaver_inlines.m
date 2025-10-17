/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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

#include "Python.h"
#include "pyobjc-api.h"
#import <ScreenSaver/ScreenSaver.h>

static PyObjC_function_map function_map[] = {
	{"SSCenteredRectInRect", (PyObjC_Function_Pointer)&SSCenteredRectInRect },
	{"SSRandomFloatBetween", (PyObjC_Function_Pointer)&SSRandomFloatBetween },
	{"SSRandomIntBetween", (PyObjC_Function_Pointer)&SSRandomIntBetween },
	{"SSRandomPointForSizeWithinRect", (PyObjC_Function_Pointer)&SSRandomPointForSizeWithinRect },
    { 0, 0 }
};

static PyMethodDef mod_methods[] = {
        { 0, 0, 0, 0 } /* sentinel */
};

PyObjC_MODULE_INIT(_inlines)
{
	PyObject* m = PyObjC_MODULE_CREATE(_inlines);
	if (!m) {
		PyObjC_INITERROR();
	}

	if (PyModule_AddObject(m, "_inline_list_", 
		PyObjC_CreateInlineTab(function_map)) < 0) {
			PyObjC_INITERROR();
	}

	PyObjC_INITDONE();
}
