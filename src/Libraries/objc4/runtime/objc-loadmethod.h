/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
/***********************************************************************
* objc-loadmethod.h
* Support for +load methods.
**********************************************************************/

#ifndef _OBJC_LOADMETHOD_H
#define _OBJC_LOADMETHOD_H

#include "objc-private.h"

__BEGIN_DECLS

extern void add_class_to_loadable_list(Class cls);
extern void add_category_to_loadable_list(Category cat);
extern void remove_class_from_loadable_list(Class cls);
extern void remove_category_from_loadable_list(Category cat);

extern void call_load_methods(void);

__END_DECLS

#endif
