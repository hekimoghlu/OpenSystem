/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
//  keychain_sync_test.c
//  sec
//
//  Created by Mitch Adler on 7/8/16.
//
//

#include "keychain_sync_test.h"

#include "secToolFileIO.h"

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <utilities/SecCFWrappers.h>
#include <utilities/SecCFRelease.h>

#import <Foundation/Foundation.h>

#include <Security/SecureObjectSync/SOSCloudCircle.h>

#import "NSFileHandle+Formatting.h"

static char boolToChars(bool val, char truechar, char falsechar) {
    return val? truechar: falsechar;
}

int
keychain_sync_test(int argc, char * const *argv)
{
    NSFileHandle *fhout = [NSFileHandle fileHandleWithStandardOutput];
    NSFileHandle *fherr = [NSFileHandle fileHandleWithStandardError];
    /*
     "Keychain Syncing test"

     */
    int result = 0;
    __block CFErrorRef cfError = NULL;

    static int verbose_flag = 0;
    bool dump_pending = false;

    static struct option long_options[] =
    {
        /* These options set a flag. */
        {"verbose",     no_argument,        &verbose_flag, 1},
        {"brief",       no_argument,        &verbose_flag, 0},
        /* These options donâ€™t set a flag.
         We distinguish them by their indices. */
        {"enabled-peer-views",      required_argument, 0, 'p'},
        {"message-pending-state",   no_argument,       0, 'm'},
        {0, 0, 0, 0}
    };
    static const char * params = "p:m";

    /* getopt_long stores the option index here. */
    int option_index = 0;

    NSArray<NSString*>* viewList = nil;

    int opt_result = 0;
    while (opt_result != -1) {
        opt_result = getopt_long (argc, argv, params, long_options, &option_index);
        switch (opt_result) {
            case 'p': {
                NSString* parameter = [NSString stringWithCString: optarg encoding:NSUTF8StringEncoding];

                viewList = [parameter componentsSeparatedByString:@","];
                
                }
                break;
            case 'm':
                dump_pending = true;
                break;
            case -1:
                break;
            default:
                return SHOW_USAGE_MESSAGE;
        }

    }

    if (viewList) {
        CFBooleanRef result = SOSCCPeersHaveViewsEnabled((__bridge CFArrayRef) viewList, &cfError);
        if (result != NULL) {
            [fhout writeFormat: @"Views: %@\n", viewList];
            [fhout writeFormat: @"Enabled on other peers: %@\n", CFBooleanGetValue(result) ? @"yes" : @"no"];
        }
    }

    if (dump_pending) {
        CFArrayRef peers = SOSCCCopyPeerPeerInfo(&cfError);
        if (peers != NULL) {
            [fhout writeFormat: @"Dumping state for %ld peers\n", CFArrayGetCount(peers)];

            CFArrayForEach(peers, ^(const void *value) {
                SOSPeerInfoRef thisPeer = (SOSPeerInfoRef) value;
                if (thisPeer) {
                    CFReleaseNull(cfError);
                    bool message = SOSCCMessageFromPeerIsPending(thisPeer, &cfError);
                    if (!message && cfError != NULL) {
                        [fherr writeFormat: @"Error from SOSCCMessageFromPeerIsPending: %@\n", cfError];
                    }
                    CFReleaseNull(cfError);
                    bool send = SOSCCSendToPeerIsPending(thisPeer, &cfError);
                    if (!message && cfError != NULL) {
                        [fherr writeFormat: @"Error from SOSCCSendToPeerIsPending: %@\n", cfError];
                    }
                    CFReleaseNull(cfError);

                    [fhout writeFormat: @"Peer: %c%c %@\n", boolToChars(message, 'M', 'm'), boolToChars(send, 'S', 's'), thisPeer];
                } else {
                    [fherr writeFormat: @"Non SOSPeerInfoRef in array: %@\n", value];
                }
            });
        }
    }

    if (cfError != NULL) {
        [fherr writeFormat: @"Error: %@\n", cfError];
    }

    return result;
}
