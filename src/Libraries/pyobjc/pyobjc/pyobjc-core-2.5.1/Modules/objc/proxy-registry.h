/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#ifndef PyObjC_PROXY_REGISTRY_H
#define PyObjC_PROXY_REGISTRY_H

int PyObjC_InitProxyRegistry(void);

int PyObjC_RegisterPythonProxy(id original, PyObject* proxy);
int PyObjC_RegisterObjCProxy(PyObject* original, id proxy);

void PyObjC_UnregisterPythonProxy(id original, PyObject* proxy);
void PyObjC_UnregisterObjCProxy(PyObject* original, id proxy);

id PyObjC_FindObjCProxy(PyObject* original);
PyObject* PyObjC_FindPythonProxy(id original);

#endif /* PyObjC_PROXY_REGISTRY_H */
