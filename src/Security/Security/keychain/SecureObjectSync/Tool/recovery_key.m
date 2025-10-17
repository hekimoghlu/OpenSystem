/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
#import <Foundation/Foundation.h>
#include "recovery_key.h"

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <utilities/SecCFWrappers.h>

#include <Security/SecureObjectSync/SOSCloudCircle.h>
#include <Security/SecureObjectSync/SOSCloudCircleInternal.h>
#include <Security/SecRecoveryKey.h>
#import <Security/SecPasswordGenerate.h>

#import <CoreCDP/CoreCDP.h>


#include "secToolFileIO.h"

int
recovery_key(int argc, char * const *argv)
{
    int ch, result = 0;
    CFErrorRef error = NULL;
    BOOL hadError = false;
    SOSLogSetOutputTo(NULL, NULL); 
    
    static struct option long_options[] =
    {
        /* These options set a flag. */
        {"recovery-string", no_argument, NULL, 'R' },
        {"generate",    required_argument, NULL, 'G'},
        {"set",         required_argument, NULL, 's'},
        {"set-and-backup", required_argument, NULL, 'b'},
        {"get",         no_argument, NULL, 'g'},
        {"clear",       no_argument, NULL, 'c'},
        {"clear-and-backup", no_argument, NULL, 'K'},
        {"follow-up",   no_argument, NULL, 'F'},
        {"verifier",   no_argument, NULL, 'V'},
        {0, 0, 0, 0}
    };
    int option_index = 0;
    
    while ((ch = getopt_long(argc, argv, "FG:Rs:b:gcKV:", long_options, &option_index)) != -1)
        switch  (ch) {
            case 'G': {
                NSError *nserror = NULL;
                NSString *testString = [NSString stringWithUTF8String:optarg];
                if(testString == nil)
                    return SHOW_USAGE_MESSAGE;
                
                SecRecoveryKey *rk = SecRKCreateRecoveryKeyWithError(testString, &nserror);
                if(rk == nil) {
                    printmsg(CFSTR("SecRKCreateRecoveryKeyWithError: %@\n"), nserror);
                    return SHOW_USAGE_MESSAGE;
                }
                NSData *publicKey = SecRKCopyBackupPublicKey(rk);
                if(publicKey == nil)
                    return SHOW_USAGE_MESSAGE;
                
                printmsg(CFSTR("example (not registered) public recovery key: %@\n"), publicKey);
                break;
            }
            case 'R': {
                NSString *testString = SecRKCreateRecoveryKeyString(NULL);
                if(testString == nil)
                    return SHOW_USAGE_MESSAGE;
                
                printmsg(CFSTR("public recovery string: %@\n"), testString);
                
                break;
            }
            case 's':
            {
                NSError *nserror = NULL;
                NSString *testString = [NSString stringWithUTF8String:optarg];
                if(testString == nil)
                    return SHOW_USAGE_MESSAGE;
                
                SecRecoveryKey *rk = SecRKCreateRecoveryKeyWithError(testString, &nserror);
                if(rk == nil) {
                    printmsg(CFSTR("SecRKCreateRecoveryKeyWithError: %@\n"), nserror);
                    return SHOW_USAGE_MESSAGE;
                }
                
                CFErrorRef cferror = NULL;
                if(!SecRKRegisterBackupPublicKey(rk, &cferror)) {
                    printmsg(CFSTR("Error from SecRKRegisterBackupPublicKey: %@\n"), cferror);
                    CFReleaseNull(cferror);
                    return SHOW_USAGE_MESSAGE;
                }
                break;
            }
            case 'b':
            {
                NSError *nserror = NULL;
                NSString *testString = [NSString stringWithUTF8String:optarg];
                if(testString == nil)
                    return SHOW_USAGE_MESSAGE;
                
                SecRecoveryKey *rk = SecRKCreateRecoveryKeyWithError(testString, &nserror);
                if(rk == nil) {
                    printmsg(CFSTR("SecRKCreateRecoveryKeyWithError: %@\n"), nserror);
                    return SHOW_USAGE_MESSAGE;
                }
                CFErrorRef copyError = NULL;
                SOSPeerInfoRef peer = SOSCCCopyMyPeerInfo(&copyError);
                if (peer) {
                    CFDataRef backupKey = SOSPeerInfoCopyBackupKey(peer);
                    if (backupKey == NULL) {
                        CFErrorRef cferr = NULL;
                        NSString *str = CFBridgingRelease(SecPasswordGenerate(kSecPasswordTypeiCloudRecovery, &cferr, NULL));
                        if (str) {
                            NSData* secret = [str dataUsingEncoding:NSUTF8StringEncoding];
                            
                            CFErrorRef registerError = NULL;
                            SOSPeerInfoRef peerInfo = SOSCCCopyMyPeerWithNewDeviceRecoverySecret((__bridge CFDataRef)secret, &registerError);
                            if (peerInfo) {
                                printmsg(CFSTR("octagon-register-recovery-key, registered backup key\n"));
                            } else {
                                printmsg(CFSTR("octagon-register-recovery-key, SOSCCCopyMyPeerWithNewDeviceRecoverySecret() failed: %@\n"), registerError);
                            }
                            CFReleaseNull(registerError);
                            CFReleaseNull(peerInfo);
                        } else {
                            printmsg(CFSTR("octagon-register-recovery-key, SecPasswordGenerate() failed: %@\n"), cferr);
                        }
                        CFReleaseNull(cferr);
                    } else {
                        printmsg(CFSTR("octagon-register-recovery-key, backup key already registered\n"));
                    }
                    CFReleaseNull(backupKey);
                    CFReleaseNull(peer);
                } else {
                    printmsg(CFSTR("octagon-register-recovery-key, SOSCCCopyMyPeerInfo() failed: %@\n"), copyError);
                }
                
                CFErrorRef cferror = NULL;
                if(!SecRKRegisterBackupPublicKey(rk, &cferror)) {
                    printmsg(CFSTR("Error from SecRKRegisterBackupPublicKey: %@\n"), cferror);
                    CFReleaseNull(cferror);
                    return SHOW_USAGE_MESSAGE;
                }
                break;
            }
            case 'g':
            {
                CFDataRef recovery_key = SOSCCCopyRecoveryPublicKey(&error);
                hadError = recovery_key == NULL;
                if(!hadError)
                    printmsg(CFSTR("recovery key: %@\n"), recovery_key);
                CFReleaseNull(recovery_key);
                break;
            }
            case 'c':
            {
                hadError = SOSCCRegisterRecoveryPublicKey(NULL, &error) != true;
                break;
            }
            case 'K':
            {
                CFErrorRef copyError = NULL;
                SOSPeerInfoRef peer = SOSCCCopyMyPeerInfo(&copyError);
                if (peer) {
                    CFDataRef backupKey = SOSPeerInfoCopyBackupKey(peer);
                    if (backupKey == NULL) {
                        CFErrorRef cferr = NULL;
                        NSString *str = CFBridgingRelease(SecPasswordGenerate(kSecPasswordTypeiCloudRecovery, &cferr, NULL));
                        if (str) {
                            NSData* secret = [str dataUsingEncoding:NSUTF8StringEncoding];
                            
                            CFErrorRef registerError = NULL;
                            SOSPeerInfoRef peerInfo = SOSCCCopyMyPeerWithNewDeviceRecoverySecret((__bridge CFDataRef)secret, &registerError);
                            if (peerInfo) {
                                secnotice("octagon-register-recovery-key", "registered backup key");
                            } else {
                                secerror("octagon-register-recovery-key, SOSCCCopyMyPeerWithNewDeviceRecoverySecret() failed: %@", registerError);
                            }
                            CFReleaseNull(registerError);
                            CFReleaseNull(peerInfo);
                        } else {
                            secerror("octagon-register-recovery-key, SecPasswordGenerate() failed: %@", cferr);
                        }
                        CFReleaseNull(cferr);
                    } else {
                        secnotice("octagon-register-recovery-key", "backup key already registered");
                    }
                    CFReleaseNull(backupKey);
                    CFReleaseNull(peer);
                } else {
                    secerror("octagon-register-recovery-key, SOSCCCopyMyPeerInfo() failed: %@", copyError);
                }
                
                CFReleaseNull(copyError);
                hadError = SOSCCRegisterRecoveryPublicKey(NULL, &error) != true;

                break;
            }
            case 'F':
            {
#if TARGET_OS_WATCH || TARGET_OS_TV
                printmsg(CFSTR("Cannot post recovery key follow ups on this platform)"));
                hadError = true;
                break;
#else
                NSError *localError = nil;

                CDPFollowUpController *cdpd = [[CDPFollowUpController alloc] init];

                CDPFollowUpContext *context = [CDPFollowUpContext contextForRecoveryKeyRepair];
                context.force = true;

                secnotice("followup", "Posting a follow up (for SOS) of type recovery key");
                [cdpd postFollowUpWithContext:context error:&localError];
                if(localError){
                    printmsg(CFSTR("Request to CoreCDP to follow up failed: %@\n"), localError);
                } else {
                    printmsg(CFSTR("CoreCDP handling follow up\n"));
                }
                break;
#endif  // TARGET_OS_WATCH || TARGET_OS_TV
            }
            case 'V': {
                NSError *localError = nil;
                NSString *testString = [NSString stringWithUTF8String:optarg];
                NSString *fileName = [NSString stringWithFormat:@"%@.plist", testString];
                if(testString == nil)
                    return SHOW_USAGE_MESSAGE;
                
                NSDictionary *ver = SecRKCopyAccountRecoveryVerifier(testString, &localError);
                if(ver == nil) {
                    printmsg(CFSTR("Failed to make verifier dictionary: %@\n"), localError);
                    return SHOW_USAGE_MESSAGE;
                }
                
                printmsg(CFSTR("Verifier Dictionary: %@\n\n"), ver);
                printmsg(CFSTR("Writing plist to %@\n"), (__bridge CFStringRef) fileName);

                [ver writeToFile:fileName atomically:YES];

                }
                break;

            case '?':
            default:
            {
                printf("%s [...options]\n", getprogname());
                for (unsigned n = 0; n < sizeof(long_options)/sizeof(long_options[0]); n++) {
                    printf("\t [-%c|--%s\n", long_options[n].val, long_options[n].name);
                }
                return SHOW_USAGE_MESSAGE;
            }
        }
    if (hadError)
        printerr(CFSTR("Error: %@\n"), error);

    return result;
}
