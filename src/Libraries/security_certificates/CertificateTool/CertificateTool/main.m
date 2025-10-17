/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
//  main.m
//  CertificateTool
//
//  Copyright (c) 2012-2015 Apple Inc. All Rights Reserved.
//

#import <Foundation/Foundation.h>
#import "CertificateToolApp.h"
#import "ValidateAsset.h"

/*
 printf("%s usage:\n", [self.app_name UTF8String]);
 printf(" [-h, --help]          			\tPrint out this help message\n");
 printf(" [-r, --roots_dir]     			\tThe full path to the directory with the certificate roots\n");
 printf(" [-k, --revoked_dir]   			\tThe full path to the directory with the revoked certificates\n");
 printf(" [-d, --distrusted_dir] 		\tThe full path to the directory with the distrusted certificates\n");
 printf(" [-a, --allowlist_dir] 		\tThe full path to the directory with the allowed certificates\n");
 printf(" [-c, --certs_dir] 				\tThe full path to the directory with the cert certificates\n");
 printf(" [-e, --ev_plist_path] 			\tThe full path to the EVRoots.plist file\n");
 printf(" [-t, --top_level_directory]	\tThe full path to the top level security_certificates directory\n");
 printf(" [-o, --output_directory]       \tThe full path to the directory to write out the results\n");
 printf("\n");
 */

int main(int argc, const char * argv[])
{
    
/* ============================================================
    This section is only used to help debug this tool
    Uncommenting out the HARDCODE line will allow for testing 
    this tool with having to run the BuildiOSAsset script
   ============================================================ */
//#define HARDCODE 1
    
#ifdef HARDCODE
    
    const char* myArgv[] =
    {
        "foo",
        "--top_level_directory",
        "/Volumes/Data/RestoreStuff/Branches/PR-14030167/security/certificates/CertificateTool/..",
        "--output_directory",
        "~/BuiltAssets"
    };
    
    int myArgc = (sizeof(myArgv) / sizeof(const char*));
    
    argc = myArgc;
    argv = myArgv;
#endif  // HARDCODE
    
    
    @autoreleasepool
    {
        CertificateToolApp* app = [[CertificateToolApp alloc] init:argc withArguments:argv];
        if (![app processCertificates])
        {
            NSLog(@"Could not process the certificate directories");
            return -1;
        }
        
        if (![app outputPlistsToDirectory])
        {
            NSLog(@"Could not output the plists");
            return -1;
        }
                
        if (![app createManifest])
        {
            NSLog(@"Could not create the manifest");
            return -1;
        }
        
        
    }
    return 0;
}

