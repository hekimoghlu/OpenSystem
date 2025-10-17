/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#include <mrc/object.h>

#include "mdns_obj.h"
#include "mrc_object_internal.h"
#include "mdns_strict.h"

//======================================================================================================================
// MARK: - Object Public Methods

mdns_kind_t
mrc_get_kind(const mrc_object_t me)
{
	return mdns_obj_get_kind(me);
}

//======================================================================================================================

void
mrc_retain(const mrc_object_t me)
{
	mdns_obj_retain(me);
}

//======================================================================================================================

void
mrc_release(const mrc_object_t me)
{
	mdns_obj_release(me);
}
