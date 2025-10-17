/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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
//
// handleobject - give an object a process-global unique handle
//
#ifndef _H_HANDLEOBJECT
#define _H_HANDLEOBJECT

#include <Security/cssm.h>
#include <security_cdsa_utilities/handletemplates.h>

//
// definitions kept here so other code doesn't have to modify #includes
//

namespace Security
{

typedef TypedHandle<CSSM_HANDLE> HandledObject;

typedef MappingHandle<CSSM_HANDLE> HandleObject;

} // end namespace Security

#endif //_H_HANDLEOBJECT
