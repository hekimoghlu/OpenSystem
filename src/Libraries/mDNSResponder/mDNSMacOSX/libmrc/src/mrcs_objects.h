/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
#ifndef MRCS_OBJECTS_H
#define MRCS_OBJECTS_H

#include "mdns_objects.h"

#define MRCS_OBJECT_SUBKIND_DEFINE_ABSTRACT(NAME)	MDNS_OBJ_SUBKIND_DEFINE_ABSTRACT(mrcs_ ## NAME)
#define MRCS_OBJECT_SUBKIND_DEFINE(NAME)			MDNS_OBJ_SUBKIND_DEFINE(mrcs_ ## NAME)
#define MRCS_OBJECT_SUBKIND_DEFINE_FULL(NAME)		MDNS_OBJ_SUBKIND_DEFINE_FULL(mrcs_ ## NAME)

#define MRCS_BASE_CHECK(NAME, SUPER)	MDNS_OBJ_BASE_CHECK(mrcs_ ## NAME, mrcs_ ## SUPER)

#define MRCS_CLASS(NAME)		MDNS_OBJ_CLASS(mrcs_ ## NAME)
#define MRCS_CLASS_DECL(NAME)	MDNS_OBJ_CLASS_DECL(mrcs_ ## NAME)

#endif	// MRCS_OBJECTS_H
