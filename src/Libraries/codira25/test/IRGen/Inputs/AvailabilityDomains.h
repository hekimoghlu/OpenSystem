/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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

#include <feature-availability.h>

static struct __AvailabilityDomain __EnabledDomain __attribute__((
    availability_domain(EnabledDomain))) = {__AVAILABILITY_DOMAIN_ENABLED, 0};

static struct __AvailabilityDomain __DisabledDomain __attribute__((
    availability_domain(DisabledDomain))) = {__AVAILABILITY_DOMAIN_DISABLED, 0};

int dynamic_domain_pred();

static struct __AvailabilityDomain __DynamicDomain
    __attribute__((availability_domain(DynamicDomain))) = {
        __AVAILABILITY_DOMAIN_DYNAMIC, dynamic_domain_pred};

#define AVAIL 0
#define UNAVAIL 1

__attribute__((availability(domain : EnabledDomain, AVAIL))) void
available_in_enabled_domain(void);

__attribute__((availability(domain : EnabledDomain, UNAVAIL))) void
unavailable_in_enabled_domain(void);

__attribute__((availability(domain : DisabledDomain, AVAIL))) void
available_in_disabled_domain(void);

__attribute__((availability(domain : DisabledDomain, UNAVAIL))) void
unavailable_in_disabled_domain(void);

__attribute__((availability(domain : DynamicDomain, AVAIL))) void
available_in_dynamic_domain(void);

__attribute__((availability(domain : DynamicDomain, UNAVAIL))) void
unavailable_in_dynamic_domain(void);
