/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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
#pragma once

#include <stdbool.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

// TODO: libdl has several private extensions, but they have not all moved into a standard
// private header.

/**
 * Set whether to load libraries in app compat mode.
 *
 * Any library which is not 16 KB aligned on a 4 KB aligned
 * will be loaded in a special mode, which may load some R-only
 * code as RW, in order to increase compatibility.
 *
 * \param enable_app_compat whether the mode is enabled for additional
 *     library loads.
 */
void android_set_16kb_appcompat_mode(bool enable_app_compat);

__END_DECLS
