/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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

#ifndef _NPY_PRIVATE__DATETIME_BUSDAY_H_
#define _NPY_PRIVATE__DATETIME_BUSDAY_H_

/*
 * This is the 'busday_offset' function exposed for calling
 * from Python.
 */
NPY_NO_EXPORT PyObject *
array_busday_offset(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds);

/*
 * This is the 'busday_count' function exposed for calling
 * from Python.
 */
NPY_NO_EXPORT PyObject *
array_busday_count(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds);

/*
 * This is the 'is_busday' function exposed for calling
 * from Python.
 */
NPY_NO_EXPORT PyObject *
array_is_busday(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds);

#endif
