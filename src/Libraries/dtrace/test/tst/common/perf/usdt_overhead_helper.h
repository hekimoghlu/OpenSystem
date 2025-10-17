/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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

#ifndef _USDT_OVERHEAD_HELPER_H_
#define _USDT_OVERHEAD_HELPER_H_
#include <usdt_overhead_helper_provider.h>

#define _CONCAT(a, b) a ## b
#define CONCAT(a, b) _CONCAT(a, b)

#define PROVIDER(i) CONCAT(CONCAT(PROVIDER, i), _TEST_PROVIDER_PROBE)();

#define PROV10(i) PROVIDER(CONCAT(i, 0)) \
PROVIDER(CONCAT(i, 1)) \
PROVIDER(CONCAT(i, 2)) \
PROVIDER(CONCAT(i, 3)) \
PROVIDER(CONCAT(i, 4)) \
PROVIDER(CONCAT(i, 5)) \
PROVIDER(CONCAT(i, 6)) \
PROVIDER(CONCAT(i, 7)) \
PROVIDER(CONCAT(i, 8)) \
PROVIDER(CONCAT(i, 9))

#define PROV100() PROV10( ) \
PROV10(1) \
PROV10(2) \
PROV10(3) \
PROV10(4) \
PROV10(5) \
PROV10(6) \
PROV10(7) \
PROV10(8) \
PROV10(9)

#endif /* _USDT_OVERHEAD_HELPER_H_ */
