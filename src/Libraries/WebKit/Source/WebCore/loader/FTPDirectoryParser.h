/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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
/* ParseFTPList() parses lines from an FTP LIST command.
**
** Written July 2002 by Cyrus Patel <cyp@fb14.uni-mainz.de>
** with acknowledgements to squid, lynx, wget and ftpmirror.
**
** Arguments:
**   'line':       line of FTP data connection output. The line is assumed
**                 to end at the first '\0' or '\n' or '\r\n'. 
**   'state':      a structure used internally to track state between 
**                 lines. Needs to be bzero()'d at LIST begin.
**   'result':     where ParseFTPList will store the results of the parse
**                 if 'line' is not a comment and is not junk.
**
** Returns one of the following:
**    'd' - LIST line is a directory entry ('result' is valid)
**    'f' - LIST line is a file's entry ('result' is valid)
**    'l' - LIST line is a symlink's entry ('result' is valid)
**    '?' - LIST line is junk. (cwd, non-file/dir/link, etc)
**    '"' - its not a LIST line (its a "comment")
**
** It may be advisable to let the end-user see "comments" (particularly when 
** the listing results in ONLY such lines) because such a listing may be:
** - an unknown LIST format (NLST or "custom" format for example)
** - an error msg (EPERM,ENOENT,ENFILE,EMFILE,ENOTDIR,ENOTBLK,EEXDEV etc).
** - an empty directory and the 'comment' is a "total 0" line or similar.
**   (warning: a "total 0" can also mean the total size is unknown).
**
** ParseFTPList() supports all known FTP LISTing formats:
** - '/bin/ls -l' and all variants (including Hellsoft FTP for NetWare); 
** - EPLF (Easily Parsable List Format); 
** - Windows NT's default "DOS-dirstyle";
** - OS/2 basic server format LIST format;  
** - VMS (MultiNet, UCX, and CMU) LIST format (including multi-line format);
** - IBM VM/CMS, VM/ESA LIST format (two known variants);  
** - SuperTCP FTP Server for Win16 LIST format;  
** - NetManage Chameleon (NEWT) for Win16 LIST format;  
** - '/bin/dls' (two known variants, plus multi-line) LIST format;
** If there are others, then I'd like to hear about them (send me a sample).
**
** NLSTings are not supported explicitely because they cannot be machine 
** parsed consistantly: NLSTings do not have unique characteristics - even 
** the assumption that there won't be whitespace on the line does not hold
** because some nlistings have more than one filename per line and/or
** may have filenames that have spaces in them. Moreover, distinguishing
** between an error message and an NLST line would require ParseList() to
** recognize all the possible strerror() messages in the world.
*/

// This was originally Mozilla code, titled ParseFTPList.h
// Original version of this file can currently be found at: http://mxr.mozilla.org/mozilla1.8/source/netwerk/streamconv/converters/ParseFTPList.h

#pragma once

#include <wtf/StdLibExtras.h>
#include <wtf/text/WTFString.h>

#include <time.h>

#define SUPPORT_LSL  /* Support for /bin/ls -l and dozens of variations therof */
#define SUPPORT_DLS  /* Support for /bin/dls format (very, Very, VERY rare) */
#define SUPPORT_EPLF /* Support for Extraordinarily Pathetic List Format */
#define SUPPORT_DOS  /* Support for WinNT server in 'site dirstyle' dos */
#define SUPPORT_VMS  /* Support for VMS (all: MultiNet, UCX, CMU-IP) */
#define SUPPORT_CMS  /* Support for IBM VM/CMS,VM/ESA (z/VM and LISTING forms) */
#define SUPPORT_OS2  /* Support for IBM TCP/IP for OS/2 - FTP Server */
#define SUPPORT_W16  /* Support for win16 hosts: SuperTCP or NetManage Chameleon */

namespace WebCore {

typedef struct tm FTPTime;

struct ListState {    
    ListState()
        : now(0)
        , listStyle(0)
        , parsedOne(false)
        , carryBufferLength(0)
        , numLines(0)
    { 
        zeroBytes(nowFTPTime);
    }
    
    double      now;               /* needed for year determination */
    FTPTime     nowFTPTime;
    char        listStyle;         /* LISTing style */
    bool        parsedOne;         /* returned anything yet? */
    std::array<LChar, 84> carryBuffer;   /* for VMS multiline */
    int         carryBufferLength; /* length of name in carry_buf */
    int64_t     numLines;          /* number of lines seen */
};

enum FTPEntryType {
    FTPDirectoryEntry,
    FTPFileEntry,
    FTPLinkEntry,
    FTPMiscEntry,
    FTPJunkEntry
};

struct ListResult
{
    ListResult()
    { 
        clear();
    }
    
    void clear()
    {
        valid = false;
        type = FTPJunkEntry;
        filename = { };
        linkname = { };
        fileSize = { };
        caseSensitive = false;
        zeroBytes(modifiedTime);
    }
    
    bool valid;
    FTPEntryType type;        
    
    std::span<LChar> filename;
    std::span<LChar> linkname;
    
    String fileSize;      
    FTPTime modifiedTime; 
    bool caseSensitive; // file system is definitely case insensitive
};

FTPEntryType parseOneFTPLine(std::span<LChar> inputLine, ListState&, ListResult&);
                 
} // namespace WebCore
