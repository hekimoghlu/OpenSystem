/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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
/*
 * @OSF_COPYRIGHT@
 */

#ifndef _MACH_TASK_LEDGER_H_
#define _MACH_TASK_LEDGER_H_

/*
 * Evolving and likely to change.
 */

/*
 * Definitions for task ledger line items
 */
#define ITEM_THREADS            0       /* number of threads	*/
#define ITEM_TASKS              1       /* number of tasks	*/

#define ITEM_VM                 2       /* virtual space (bytes)*/

#define LEDGER_N_ITEMS          3       /* Total line items	*/

#define LEDGER_UNLIMITED        0       /* ignored item.maximum	*/

#endif  /* _MACH_TASK_LEDGER_H_ */
