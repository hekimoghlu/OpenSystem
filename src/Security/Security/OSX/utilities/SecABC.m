/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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

//
//  SecABC.m
//  Security
//

#import <SoftLinking/SoftLinking.h>
#import <os/log.h>

/*
 * This is using soft linking since we need upward (to workaround BNI build dependencies)
 * and weak linking since SymptomDiagnosticReporter is not available on base and darwinOS.
 */
#if ABC_BUGCAPTURE
#import <SymptomDiagnosticReporter/SDRDiagnosticReporter.h>

SOFT_LINK_OPTIONAL_FRAMEWORK(PrivateFrameworks, SymptomDiagnosticReporter);
SOFT_LINK_CLASS(SymptomDiagnosticReporter, SDRDiagnosticReporter);

#endif

#import "SecABC.h"


void SecABCTrigger(CFStringRef type,
                   CFStringRef subtype,
                   CFStringRef subtypeContext,
                   CFDictionaryRef payload)
{
    [SecABC triggerAutoBugCaptureWithType:(__bridge NSString *)type
                                  subType:(__bridge NSString *)subtype
                           subtypeContext:(__bridge NSString *)subtypeContext
                                   domain:@"com.apple.security.keychain"
                                   events:nil
                                  payload:(__bridge NSDictionary *)payload
                          detectedProcess:nil];
}


@implementation SecABC

+ (void)triggerAutoBugCaptureWithType:(NSString *)type
                              subType:(NSString *)subType
{
    [self triggerAutoBugCaptureWithType: type
                                subType: subType
                         subtypeContext: nil
                                 domain: @"com.apple.security.keychain"
                                 events: nil
                                payload: nil
                        detectedProcess: nil];
}


+ (void)triggerAutoBugCaptureWithType:(NSString *)type
                              subType:(NSString *)subType
                       subtypeContext:(NSString * _Nullable)subtypeContext
                               domain:(NSString *)domain
                               events:(NSArray * _Nullable)events
                              payload:(NSDictionary * _Nullable)payload
                      detectedProcess:(NSString * _Nullable)process
{
#if ABC_BUGCAPTURE
    os_log_info(OS_LOG_DEFAULT, "TriggerABC for %{public}@/%{public}@/%{public}@",
                type, subType, subtypeContext);

    // no ABC on darwinos
    Class sdrDiagReporter = getSDRDiagnosticReporterClass();
    if (sdrDiagReporter == nil) {
        return;
    }

    SDRDiagnosticReporter *diagnosticReporter = [[sdrDiagReporter alloc] init];
    NSMutableDictionary *signature = [diagnosticReporter signatureWithDomain:domain
                                                                        type:type
                                                                     subType:subType
                                                              subtypeContext:subtypeContext
                                                             detectedProcess:process?:[[NSProcessInfo processInfo] processName]
                                                      triggerThresholdValues:nil];
    if (signature == NULL) {
        os_log_error(OS_LOG_DEFAULT, "TriggerABC signature generation failed");
        return;
    }

    (void)[diagnosticReporter snapshotWithSignature:signature
                                              delay:5.0
                                             events:events
                                            payload:payload
                                            actions:nil
                                              reply:^void(NSDictionary *response)
    {
        os_log_info(OS_LOG_DEFAULT,
                    "Received response from Diagnostic Reporter - %{public}@/%{public}@/%{public}@: %{public}@",
                    type, subType, subtypeContext, response);
    }];
#endif
}

@end
