/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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

#ifndef OBJC_CLASS_BUILDER
#define OBJC_CLASS_BUILDER

extern PyObject* PyObjC_class_setup_hook;

Class PyObjCClass_BuildClass(
		Class super_class,  
		PyObject* protocols,
		char* name, 
		PyObject* class_dict,
		PyObject* meta_dict,
		PyObject* hiddenSelectors,
		PyObject* hiddenClassSelectors);

int PyObjCClass_UnbuildClass(Class new_class);
int PyObjCClass_FinishClass(Class objc_class);
void PyObjC_RemoveInternalTypeCodes(char* buf);

#endif /* OBJC_CLASS_BUILDER */
