/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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
#include <security_cdsa_utilities/handletemplates_defs.h>
#include <Security/cssm.h>
#include <stdint.h>

namespace Security
{

// 
// Instantiate the explicit MappingHandle subclasses.  If there start to be
// a lot of these, break this into multiple .cpp files so useless classes
// aren't linked in everywhere.  
//

template struct TypedHandle<CSSM_HANDLE>;        // HandledObject

template class MappingHandle<CSSM_HANDLE>;      // HandleObject

template class MappingHandle<uint32_t>;         // U32HandleObject

}
