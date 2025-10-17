/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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

#ifndef PyOBJC_BLOCK_SUPPORT_H
#define PyOBJC_BLOCK_SUPPORT_H

typedef void (*_block_func_ptr)(void*, ...);
extern _block_func_ptr PyObjCBlock_GetFunction(void* block);
extern const char* PyObjCBlock_GetSignature(void* _block);
extern void* PyObjCBlock_Create(PyObjCMethodSignature* signature, PyObject* callable);
extern void PyObjCBlock_Release(void* _block);
extern int PyObjCBlock_Setup(void);
extern PyObject* PyObjCBlock_Call(PyObject* self, PyObject* args);


#endif /* PyOBJC_BLOCK_SUPPORT_H */
