/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#import "FakeXCTest.h"
#import <getopt.h>
#import "json_support.h"


extern char g_test_url1[1024];
extern char g_test_url2[1024];
extern char g_test_url3[1024];

CFMutableDictionaryRef json_dict = NULL;
CFMutableDictionaryRef test_mdata_dict = NULL;

int json = 0;
int list_tests_only = 0;
extern int list_tests_with_mdata;

static void
usage(void)
{
    fprintf (stderr, "usage: smbclient_test [-f JSON] [-l testName] [-e] [-o output_file] -1 URL1 -2 URL2 -3 SMB1-URL [-h] \n\
             -f%s Print info in the provided format. Supported formats: JSON \n\
             -l%s Limit run to only testName \n\
             -e%s List out all testNames (doesnt require -1, -2, -3 args) \n\
             -a%s List out all testNames with more info in JSON format (doesnt require -1, -2, -3 args) \n\
             -o%s Filename to write JSON output to \n\
             -1%s Used for single user testing. Format of smb://domain;user:password@server/share \n\
             -2%s Has to be a different user to the same server/share used in URL1. Format of smb://domain;user2:password@server/share \n\
             -3%s Has to be a CIFS URL to a server that supports SMB v1. Format of cifs://domain;user:password@server/share   \n\
             \n",
             ",--format  ",
             ",--limit   ",
             ",--list    ",
             ",--all    ",
             ",--outfile ",
             ",--url1    ",
             ",--url2    ",
             ",--url3    "
             );
    exit(EINVAL);
}

int
main(int argc, char **argv) {
    int result = 1;
    int ch;
    char *output_file_path = NULL;
    FILE *fd = NULL;

    static struct option longopts[] = {
        { "format",     required_argument,      NULL,           'f' },
        { "limit",      required_argument,      NULL,           'l' },
        { "list",       no_argument,            NULL,           'e' },
        { "all",        no_argument,            NULL,           'a' },
        { "outfile",    required_argument,      NULL,           'o' },
        { "url1",       required_argument,      NULL,           '1' },
        { "url2",       required_argument,      NULL,           '2' },
        { "url3",       required_argument,      NULL,           '3' },
        { "help",       no_argument,            NULL,           'h' },
        { NULL,         0,                      NULL,           0   }
    };

    optind = 0;
    while ((ch = getopt_long(argc, argv, "aef:l:o:1:2:3:h", longopts, NULL)) != -1) {
        switch (ch) {
            case 'a':
                list_tests_with_mdata = 1;
                break;
            case 'e':
                list_tests_only = 1;
                break;
            case 'f':
                if (strcasecmp(optarg, "json") == 0) {
                    json = 1;
                }
                else {
                    usage();
                }
                break;
            case 'l':
                [XCTest setLimit:optarg];
                break;
            case 'o':
                output_file_path = optarg;
                break;
            case '1':
                strlcpy(g_test_url1, optarg, sizeof(g_test_url1));
                break;
            case '2':
                strlcpy(g_test_url2, optarg, sizeof(g_test_url2));
                break;
            case '3':
                strlcpy(g_test_url3, optarg, sizeof(g_test_url2));
                break;
            case 'h':
            default:
                usage();
                break;
        }
    }

    if ((list_tests_only == 0) &&
        (list_tests_with_mdata == 0)) {
        /* Check for required arguments */
        if (strnlen(g_test_url1, sizeof(g_test_url1)) == 0) {
            fprintf(stderr, "URL1 is null \n");
            usage();
        }

        if (strnlen(g_test_url2, sizeof(g_test_url2)) == 0) {
            fprintf(stderr, "URL2 is null \n");
            usage();
        }

        if (strnlen(g_test_url3, sizeof(g_test_url3)) == 0) {
            fprintf(stderr, "URL3 is null \n");
            usage();
        }
    }
    
    json_dict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
                                          &kCFTypeDictionaryKeyCallBacks,
                                          &kCFTypeDictionaryValueCallBacks);
    if (json_dict == NULL) {
        fprintf(stderr, "*** %s: CFDictionaryCreateMutable failed\n",
                __FUNCTION__);
        return( ENOMEM );
    }

    if ((json == 0) && (list_tests_only == 0) && (list_tests_with_mdata == 0)) {
        /* Not using JSON */
        printf("URL1: <%s> \n", g_test_url1);
        printf("URL2: <%s> \n", g_test_url2);
        printf("SMB1 URL3: <%s> \n", g_test_url3);
    }

    @autoreleasepool {
        result = [XCTest runTests];
    }

    // If redirected stdout, then close it here before writing JSON
    if ( fd != NULL) {
        fclose(fd);
    }

    if ((json == 1) || (list_tests_with_mdata == 1)) {
        // Dont CFRelease the dictionary after JSON printing
        json_print_cf_object(json_dict, output_file_path);
        printf("\n");
    }

    return result;
}

