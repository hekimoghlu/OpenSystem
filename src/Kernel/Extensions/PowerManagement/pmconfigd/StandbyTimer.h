/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
#ifndef StandbyTimer_h
#define StandbyTimer_h


__private_extern__ void evaluateAdaptiveStandby(void);
__private_extern__ void setStandbyTimer(void);
__private_extern__ void standbyTimer_prime(void);
__private_extern__ CFAbsoluteTime getWakeFromStandbyTime(void);
__private_extern__ void setInactivityWindow(xpc_object_t remoteConnection, xpc_object_t msg);
__private_extern__ int64_t getDeltaToStandby(void);
__private_extern__ void resetDeltaToStandby(void);
__private_extern__ void refreshStandbyInactivityPrediction(void);



#endif
