/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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
#ifndef PowerManagement_Platform_h
#define PowerManagement_Platform_h
#include "XCTest_FunctionDefinitions.h"
#include "PrivateLib.h"


#define kTCPKeepAliveExpireSecs (12*60*60) // 12 hours

typedef enum {
    kNotSupported = 0,
    kActive,
    kInactive,
} tcpKeepAliveStates_et;


typedef struct {
    long                overrideSec;
    tcpKeepAliveStates_et   state;
    XCT_UNSAFE_UNRETAINED dispatch_source_t   expiration;
    CFAbsoluteTime      ts_turnoff; // Time at which Keep Aive will be turned off
} TCPKeepAliveStruct;

/*! getTCPKeepAliveState
 *  
 *  @param buf      Upon return, this buffer will contain a string either "active",
 *                  if TCPKeepAlive is active;
 *                  or "inactive: <reasons>" with the reason that it's inactive.
 *                      inactive: expired, quota
 *                  or "unsupported"
 *  @result         Returns a state value from enum tcpKeepAliveStates_et
 */
__private_extern__ tcpKeepAliveStates_et  getTCPKeepAliveState(char *buf, int buflen, bool quiet);
__private_extern__ long getTCPKeepAliveOverrideSec(void);
__private_extern__ void setTCPKeepAliveOverrideSec(long value);

__private_extern__ void startTCPKeepAliveExpTimer(void);
__private_extern__ void cancelTCPKeepAliveExpTimer(void);
__private_extern__ CFTimeInterval getTcpkaTurnOffTime(void);

__private_extern__ void enableTCPKeepAlive(void);
__private_extern__ void disableTCPKeepAlive(void);
__private_extern__ void evalTcpkaForPSChange(int pwrSrc);


__private_extern__ void setPushConnectionState(bool active);
__private_extern__ bool getPushConnectionState(void);
__private_extern__ bool getWakeOnLanState(void);
#endif
