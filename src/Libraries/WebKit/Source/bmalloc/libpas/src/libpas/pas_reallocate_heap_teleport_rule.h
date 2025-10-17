/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
#ifndef PAS_REALLOCATE_HEAP_TELEPORT_RULE_H
#define PAS_REALLOCATE_HEAP_TELEPORT_RULE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_reallocate_heap_teleport_rule {
    pas_reallocate_allow_heap_teleport,
    pas_reallocate_disallow_heap_teleport
};

typedef enum pas_reallocate_heap_teleport_rule pas_reallocate_heap_teleport_rule;

PAS_END_EXTERN_C;

#endif /* PAS_REALLOCATE_HEAP_TELEPORT_RULE_H */

