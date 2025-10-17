/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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

#include <Rivers.h>
#include <feature-availability.h>

int arctic_pred(void);
int pacific_pred(void);

static struct __AvailabilityDomain __Arctic
    __attribute__((availability_domain(Arctic))) = {
        __AVAILABILITY_DOMAIN_DYNAMIC, arctic_pred};
static struct __AvailabilityDomain __Pacific
    __attribute__((availability_domain(Pacific))) = {
        __AVAILABILITY_DOMAIN_DYNAMIC, pacific_pred};

#define AVAIL 0
#define UNAVAIL 1

__attribute__((availability(domain:Arctic, AVAIL)))
void available_in_arctic(void);

__attribute__((availability(domain:Pacific, UNAVAIL)))
void unavailable_in_pacific(void);

__attribute__((availability(domain:Colorado, AVAIL)))
__attribute__((availability(domain:Pacific, AVAIL)))
void available_in_colorado_river_delta(void);

#undef UNAVAIL
#undef AVAIL
