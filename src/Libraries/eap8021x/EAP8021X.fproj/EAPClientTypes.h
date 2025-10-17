/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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
 * Modification History
 *
 * November 25, 2002	Dieter Siegmund (dieter@apple.com)
 * - created
 */

#ifndef _EAP802_1X_EAPCLIENTTYPES_H
#define _EAP802_1X_EAPCLIENTTYPES_H

#include <stdint.h>

enum {
    /* EAPClient specific errors */
    kEAPClientStatusOK = 0,
    kEAPClientStatusFailed = 1,
    kEAPClientStatusAllocationFailed = 2,
    kEAPClientStatusUserInputRequired = 3,
    kEAPClientStatusConfigurationInvalid = 4,
    kEAPClientStatusProtocolNotSupported = 5,
    kEAPClientStatusServerCertificateNotTrusted = 6,
    kEAPClientStatusInnerProtocolNotSupported = 7,
    kEAPClientStatusInternalError = 8,
    kEAPClientStatusUserCancelledAuthentication = 9,
    kEAPClientStatusUnknownRootCertificate = 10,
    kEAPClientStatusNoRootCertificate = 11,
    kEAPClientStatusCertificateExpired = 12,
    kEAPClientStatusCertificateNotYetValid = 13,
    kEAPClientStatusCertificateRequiresConfirmation = 14,
    kEAPClientStatusUserInputNotPossible = 15,
    kEAPClientStatusResourceUnavailable = 16,
    kEAPClientStatusProtocolError = 17,
    kEAPClientStatusAuthenticationStalled = 18,
    kEAPClientStatusIdentityDecryptionError = 19,
    kEAPClientStatusOtherInputRequired = 20,

    /* domain specific errors */
    kEAPClientStatusDomainSpecificErrorStart = 1000,
    kEAPClientStatusErrnoError = 1000,		/* errno error */
    kEAPClientStatusSecurityError = 1001,	/* Security framework error */
    kEAPClientStatusPluginSpecificError = 1002,	/* plug-in specific error */
};
typedef int32_t EAPClientStatus;

typedef int32_t EAPClientDomainSpecificError;

#endif /* _EAP8021X_EAPCLIENTTYPES_H */
