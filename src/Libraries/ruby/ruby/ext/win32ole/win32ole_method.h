/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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

#ifndef WIN32OLE_METHOD_H
#define WIN32OLE_METHOD_H 1

struct olemethoddata {
    ITypeInfo *pOwnerTypeInfo;
    ITypeInfo *pTypeInfo;
    UINT index;
};

extern VALUE cWIN32OLE_METHOD;
VALUE folemethod_s_allocate(VALUE klass);
VALUE ole_methods_from_typeinfo(ITypeInfo *pTypeInfo, int mask);
VALUE create_win32ole_method(ITypeInfo *pTypeInfo, VALUE name);
struct olemethoddata *olemethod_data_get_struct(VALUE obj);
void Init_win32ole_method(void);
#endif
