/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
//  secViewDisplay.c
//  sec
//
//
//

#include "secViewDisplay.h"
#include "secToolFileIO.h"

#include <Security/SecureObjectSync/SOSCloudCircle.h>
#include <Security/SecureObjectSync/SOSCloudCircleInternal.h>
#include <Security/SecureObjectSync/SOSViews.h>


static struct foo {
    const char *name;
    const CFStringRef *viewspec;
} string2View[] = {
    { "keychain", &kSOSViewKeychainV0 },
#undef DOVIEWMACRO
#define DOVIEWMACRO(VIEWNAME, DEFSTRING, CMDSTRING, SYSTEM, DEFAULTSETTING, INITIALSYNCSETTING, ALWAYSONSETTING, BACKUPSETTING, V0SETTING) \
    { CMDSTRING, &k##SYSTEM##View##VIEWNAME, },
#include "keychain/SecureObjectSync/ViewList.list"
};

static CFStringRef convertStringToView(char *viewname) {
    unsigned n;
    
    for (n = 0; n < sizeof(string2View)/sizeof(string2View[0]); n++) {
        if (strcmp(string2View[n].name, viewname) == 0)
            return *string2View[n].viewspec;
    }
    
    // Leak this, since it's a getter.
    return CFStringCreateWithCString(kCFAllocatorDefault, viewname, kCFStringEncodingUTF8);
}

static CFStringRef convertViewReturnCodeToString(SOSViewActionCode ac) {
    CFStringRef retval = NULL;
    switch(ac) {
        case kSOSCCGeneralViewError:
            retval = CFSTR("General Error"); break;
        case kSOSCCViewMember:
            retval = CFSTR("Is Member of View"); break;
        case kSOSCCViewNotMember:
            retval = CFSTR("Is Not Member of View"); break;
        case kSOSCCViewNotQualified:
            retval = CFSTR("Is not qualified for View"); break;
        case kSOSCCNoSuchView:
            retval = CFSTR("No Such View"); break;
    }
    return retval;
}

bool viewcmd(char *itemName, CFErrorRef *err) {
    char *cmd, *viewname;
    SOSViewActionCode ac = kSOSCCViewQuery;
    CFStringRef viewspec;
    
    viewname = strchr(itemName, ':');
    if(viewname == NULL) return false;
    *viewname = 0;
    viewname++;
    cmd = itemName;
    
    if(strcmp(cmd, "enable") == 0) {
        ac = kSOSCCViewEnable;
    } else if(strcmp(cmd, "disable") == 0) {
        ac = kSOSCCViewDisable;
    } else if(strcmp(cmd, "query") == 0) {
        ac = kSOSCCViewQuery;
    } else {
        return false;
    }
    
    if(strchr(viewname, ',') == NULL) { // original single value version
        viewspec = convertStringToView(viewname);
        if(!viewspec) return false;
        
        SOSViewResultCode rc = SOSCCView(viewspec, ac, err);
        CFStringRef resultString = convertViewReturnCodeToString(rc);
        
        printmsg(CFSTR("View Result: %@ : %@\n"), resultString, viewspec);
        return true;
    }
    
    if(ac == kSOSCCViewQuery) return false;
    
    // new multi-view version
    char *viewlist = strdup(viewname);
    char *token;
    char *tofree = viewlist;
    CFMutableSetRef viewSet = CFSetCreateMutable(NULL, 0, &kCFCopyStringSetCallBacks);
    
    while ((token = strsep(&viewlist, ",")) != NULL) {
        CFStringRef resultString = convertStringToView(token);
        CFSetAddValue(viewSet, resultString);
    }
    
    printmsg(CFSTR("viewSet provided is %@\n"), viewSet);
    
    free(tofree);
    
    bool retcode;
    if(ac == kSOSCCViewEnable) retcode = SOSCCViewSet(viewSet, NULL);
    else retcode = SOSCCViewSet(NULL, viewSet);
    
    fprintf(outFile, "SOSCCViewSet returned %s\n", (retcode)? "true": "false");
    
    return true;
}

bool listviewcmd(CFErrorRef *err) {
    unsigned n;
    
    for (n = 0; n < sizeof(string2View)/sizeof(string2View[0]); n++) {
        CFStringRef viewspec = *string2View[n].viewspec;
        
        SOSViewResultCode rc = SOSCCView(viewspec, kSOSCCViewQuery, err);
        CFStringRef resultString = convertViewReturnCodeToString(rc);
        
        printmsg(CFSTR("View Result: %@ : %@\n"), resultString, viewspec);
    };
    
    return true;
}
