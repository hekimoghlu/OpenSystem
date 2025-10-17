/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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
#ifndef _EAP8021X_EAPCLIENTCONFIGURATION_H
#define _EAP8021X_EAPCLIENTCONFIGURATION_H

/*
 * EAPClientConfiguration.h
 * - functions to handle EAPClientConfiguration validations
 */

#include <EAP8021X/EAP.h>
#include <stdbool.h>
#include <stdio.h>
#include <CoreFoundation/CFString.h>

/*
 * Function: EAPClientConfigurationCopyShareable
 *
 * Purpose:
 *   This function takes the original EAPClientConfiguration dictionary as an input and returns
 *   a dictionary with shareable EAPClientConfiguration dictionary and shareable identity information
 *   if input EAPClientConfiguration dictionary contains "TLSIdentityHandle" key.
 *   This function is meant to be called on the source device and returned dictionary should be
 *   passed to EAPClientConfigurationCopyAndImport() on destination device(e.g. HomePod).
 *
 * Returns:
 *   NULL if the input EAPClientConfiguration dictionary cannot be shared, non-NULL CFDictionaryRef otherwise.
 */
CFDictionaryRef
EAPClientConfigurationCopyShareable(CFDictionaryRef eapConfiguration);

/*
 * Function: EAPClientConfigurationCopyAndImport
 *
 * Purpose:
 *   This function takes a dictionary returned by EAPClientConfigurationCopyShareable() as an input
 *   and returns EAPClientConfiguration dictionary that can be used for EAP authentication.
 *   If input dictionary contains valid identity information dictionary then output EAPClientConfiguration
 *   dictionary will contain "TLSIdentityHandle" key.
 *   This function is meant to be called on the destination device(e.g. HomePod) and returned dictionary should be
 *   configured for EAP authentication.
 *
 * Returns:
 *   NULL if this function fails to validate the input dictionary, non-NULL CFDictionaryRef otherwise.
 */
CFDictionaryRef
EAPClientConfigurationCopyAndImport(CFDictionaryRef shareableEapConfiguration);

#endif /* _EAP8021X_EAPCLIENTCONFIGURATION_H */
