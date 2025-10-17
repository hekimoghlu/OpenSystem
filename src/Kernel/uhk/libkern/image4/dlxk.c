/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#include <libkern/libkern.h>
#include <libkern/section_keywords.h>
#include <libkern/image4/dlxk.h>

#pragma mark Module Globals
SECURITY_READ_ONLY_LATE(const image4_dlxk_interface_t *) _dlxk = NULL;

#pragma mark KPI
void
image4_dlxk_link(const image4_dlxk_interface_t *dlxk)
{
	if (_dlxk) {
		panic("image4 dlxk interface already set");
	}
	_dlxk = dlxk;
}

const image4_dlxk_interface_t *
image4_dlxk_get(image4_struct_version_t v)
{
	if (v > _dlxk->dlxk_version) {
		return NULL;
	}
	return _dlxk;
}
