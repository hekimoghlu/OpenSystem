/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR

#include <fcntl.h>

#include "SecurityCommands.h"

#include <AssertMacros.h>
#include <Security/SecItemPriv.h>

#include <utilities/SecCFWrappers.h>

#include "SecurityTool/sharedTool/readline.h"
#include "SecurityTool/sharedTool/tool_errors.h"


static int
do_keychain_import(const char *backupPath, const char *keybagPath, const char *passwordString)
{
    CFDataRef backup=NULL;
    CFDataRef keybag=NULL;
    CFDataRef password=NULL;
    bool ok=false;

    if(passwordString) {
        require(password = CFDataCreate(NULL, (UInt8 *)passwordString, strlen(passwordString)), out);
    }
    require(keybag=copyFileContents(keybagPath), out);
    require(backup=copyFileContents(backupPath), out);

    ok=_SecKeychainRestoreBackup(backup, keybag, password);

out:
    CFReleaseSafe(backup);
    CFReleaseSafe(keybag);
    CFReleaseSafe(password);

    return ok?0:1;
}

static int
do_keychain_export(const char *backupPath, const char *keybagPath, const char *passwordString)
{
    CFDataRef backup=NULL;
    CFDataRef keybag=NULL;
    CFDataRef password=NULL;
    bool ok=false;

    if (keybagPath) {
        if(passwordString) {
            require(password = CFDataCreate(NULL, (UInt8 *)passwordString, strlen(passwordString)), out);
        }
        require(keybag=copyFileContents(keybagPath), out);
        require(backup=_SecKeychainCopyBackup(keybag, password), out);
        ok=writeFileContents(backupPath, backup);
    } else {
        mode_t mode = 0644; // octal!
        int fd = open(backupPath, O_RDWR|O_CREAT|O_TRUNC, mode);
        if (fd < 0) {
            sec_error("failed to open file %s (%d) %s", backupPath, errno, strerror(errno));
            goto out;
        }
        CFErrorRef error = NULL;
        ok = _SecKeychainWriteBackupToFileDescriptor(NULL, NULL, fd, &error);
        if (!ok) {
            sec_error("error: %ld", (long)CFErrorGetCode(error));
        }
    }

out:
    CFReleaseSafe(backup);
    CFReleaseSafe(keybag);
    CFReleaseSafe(password);

    return ok?0:1;
}


int
keychain_import(int argc, char * const *argv)
{
    int ch;
    int verbose=0;
    const char *keybag=NULL;
    const char *password=NULL;

    while ((ch = getopt(argc, argv, "vk:p:")) != -1)
    {
        switch (ch)
        {
            case 'v':
                verbose++;
                break;
            case 'k':
                keybag=optarg;
                break;
            case 'p':
                password=optarg;
                break;
             default:
                return SHOW_USAGE_MESSAGE;
        }
    }

    argc -= optind;
    argv += optind;

    if(keybag==NULL) {
        sec_error("-k is required\n");
        return SHOW_USAGE_MESSAGE;
    }

    if (argc != 1) {
        sec_error("<backup> is required\n");
        return SHOW_USAGE_MESSAGE;
    }
    
    return do_keychain_import(argv[0], keybag, password);
}

int
keychain_export(int argc, char * const *argv)
{
    int ch;
    int verbose=0;
    const char *keybag=NULL;
    const char *password=NULL;

    while ((ch = getopt(argc, argv, "vk:p:")) != -1)
    {
        switch (ch)
        {
            case 'v':
                verbose++;
                break;
            case 'k':
                keybag=optarg;
                break;
            case 'p':
                password=optarg;
                break;
            default:
                return SHOW_USAGE_MESSAGE;
        }
    }

    argc -= optind;
    argv += optind;

    if (keybag == NULL && password != NULL) {
        sec_error("-k is required when -p is specified\n");
        return SHOW_USAGE_MESSAGE;
    }

    if (argc != 1) {
        sec_error("<plist> is required\n");
        return SHOW_USAGE_MESSAGE;
    }

    return do_keychain_export(argv[0], keybag, password);
}

int
keychain_backup_get_uuid(int argc, char * const *argv)
{
    // Skip subcommand
    argc--;
    argv++;

    if (argc != 1) {
        sec_error("<plist> is required\n");
        return SHOW_USAGE_MESSAGE;
    }

    const char* const backupPath = argv[0];
    int fd = open(backupPath, O_RDWR);
    if (fd < 0) {
        sec_error("failed to open file %s (%d) %s", backupPath, errno, strerror(errno));
        return 1;
    }
    CFErrorRef error = NULL;
    CFStringRef uuidStr = _SecKeychainCopyKeybagUUIDFromFileDescriptor(fd, &error);
    if (!uuidStr) {
        sec_error("error: %ld", (long)CFErrorGetCode(error));
        return 1;
    }

    printf("%s\n", CFStringGetCStringPtr(uuidStr, kCFStringEncodingUTF8));
    CFReleaseNull(uuidStr);
    return 0;
}

#endif /* TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR */
