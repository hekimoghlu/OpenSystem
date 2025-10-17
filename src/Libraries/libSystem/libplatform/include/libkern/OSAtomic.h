/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
#ifndef _OSATOMIC_H_
#define _OSATOMIC_H_

/*! @header
 * These are deprecated legacy interfaces for atomic and synchronization
 * operations.
 *
 * Define OSATOMIC_USE_INLINED=1 to get inline implementations of the
 * OSAtomic interfaces in terms of the <stdatomic.h> primitives.
 *
 * Define OSSPINLOCK_USE_INLINED=1 to get inline implementations of the
 * OSSpinLock interfaces in terms of the <os/lock.h> primitives.
 *
 * These are intended as a transition convenience, direct use of those
 * primitives should be preferred.
 */

#include <sys/cdefs.h>

#include "OSAtomicDeprecated.h"
#include "OSSpinLockDeprecated.h"
#include "OSAtomicQueue.h"

#endif /* _OSATOMIC_H_ */
