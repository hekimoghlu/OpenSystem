/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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

#ifndef _NPY_UCSNARROW_H_
#define _NPY_UCSNARROW_H_

NPY_NO_EXPORT int
PyUCS2Buffer_FromUCS4(Py_UNICODE *ucs2, npy_ucs4 *ucs4, int ucs4length);

NPY_NO_EXPORT int
PyUCS2Buffer_AsUCS4(Py_UNICODE *ucs2, npy_ucs4 *ucs4, int ucs2len, int ucs4len);

NPY_NO_EXPORT PyUnicodeObject *
PyUnicode_FromUCS4(char *src, Py_ssize_t size, int swap, int align);

#endif
