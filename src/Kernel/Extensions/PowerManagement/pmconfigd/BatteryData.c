/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
 * Copyright (c) 2017 Apple Computer, Inc.  All rights reserved.
 *
 */
#include <CoreFoundation/CoreFoundation.h>
#include <CoreFoundation/CFPreferences_Private.h>
#include <CoreFoundation/CFXPCBridge.h>
#include <IOKit/pwr_mgt/IOPMLibPrivate.h>
#include <IOKit/pwr_mgt/IOPMPrivate.h>
#include <IOKit/IOReturn.h>
#include <IOKit/pwr_mgt/IOPM.h>
#include <IOKit/pwr_mgt/powermanagement_mig.h>
#include <battery/battery.h>
#include "BatteryTimeRemaining.h"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

