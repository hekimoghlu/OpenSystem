/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
#ifndef PyObjC_FSSPEC_H
#define PyObjC_FSSPEC_H

#define IS_FSSPEC(typestr) \
	(strncmp(typestr, @encode(FSSpec), sizeof(@encode(FSSpec))-1) == 0)

extern int PyObjC_encode_fsspec(PyObject*, void*);
extern PyObject* PyObjC_decode_fsspec(void*);

extern PyTypeObject PyObjC_FSSpecType;
#define PyObjC_FSSpecCheck(value) \
	PyObject_TypeCheck(value, &PyObjC_FSSpecType)

#endif /* PyObjC_FSSPEC_H */
