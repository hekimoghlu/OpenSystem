/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
 *  AuthorizationTrampolinePriv.h -- Authorization defines for communication with
 *  authtrampoline.
 *  
 */

#ifndef _SECURITY_AUTHORIZATIONTRAMPOLINEPRIV_H_
#define _SECURITY_AUTHORIZATIONTRAMPOLINEPRIV_H_

#define XPC_REQUEST_CREATE_PROCESS "createp"
#define XPC_REPLY_MSG              "rpl"
#define XPC_EVENT_MSG              "evt"
#define XPC_REQUEST_ID             "req"

#define XPC_EVENT_TYPE             "evtt"
#define XPC_EVENT_TYPE_CHILDEND    "ce"

#define PARAM_TOOL_PATH            "tool"   // path to the executable
#define PARAM_TOOL_PARAMS          "params" // parameters passed to the executable
#define PARAM_ENV                  "env"    // environment
#define PARAM_CWD                  "cwd"    // current working directory
#define PARAM_EUID                 "requid" // uid under which executable should be running
#define PARAM_AUTHREF              "auth"   // authorization
#define PARAM_EXITCODE             "ec"     // exit code of that tool
#define PARAM_STDIN                "in"     // stdin FD
#define PARAM_STDOUT               "out"    // stdout FD
#define PARAM_STDERR               "err"    // stderr FD
#define PARAM_DATA                 "data"   // data to be written to the executable's stdin
#define PARAM_CHILDEND_NEEDED      "cen"    // indicates client needs to be notified when executable finishes

#define RETVAL_STATUS              "status"
#define RETVAL_CHILD_PID           "cpid"


#endif /* !_SECURITY_AUTHORIZATIONTRAMPOLINEPRIV_H_ */
