/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
OPT_LONGONLY("help", 0, 0,
    "                 display this help and exit\n", OPT_HELP)
OPT('v', "version", 0, 0,
    "              output version information and exit\n")
OPT('m', "magic-file", 1, 0,
    " LIST      use LIST as a colon-separated list of magic\n"
    "                               number files\n")
OPT_UNIX03('M', 0, " LIST                    use LIST as a colon-separated list of magic\n"
    "                               number files in place of default\n")
OPT('z', "uncompress", 0, 0,
    "           try to look inside compressed files\n")
OPT('Z', "uncompress-noreport", 0, 0,
    "  only print the contents of compressed files\n")
OPT('b', "brief", 0, 0,
    "                do not prepend filenames to output lines\n")
OPT('c', "checking-printout", 0, 0,
    "    print the parsed form of the magic file, use in\n"
    "                               conjunction with -m to debug a new magic file\n"
    "                               before installing it\n")
OPT_UNIX03('d', 0, "                         use default magic file\n")
OPT('e', "exclude", 1, 0,
    " TEST         exclude TEST from the list of test to be\n"
    "                               performed for file. Valid tests are:\n"
    "                               %e\n")
OPT_LONGONLY("exclude-quiet", 1, 0,
    " TEST         like exclude, but ignore unknown tests\n", OPT_EXCLUDE_QUIET)
OPT('f', "files-from", 1, 0,
    " FILE      read the filenames to be examined from FILE\n")
OPT('F', "separator", 1, 0,
    " STRING     use string as separator instead of `:'\n")
OPT_UNIX03('i', 0, "                         do not further classify regular files\n")
OPT('I', "mime", 0, 0,
    "                 output MIME type strings (--mime-type and\n"
    "                               --mime-encoding)\n")
OPT_LONGONLY("extension", 0, 0,
    "            output a slash-separated list of extensions\n", OPT_EXTENSIONS)
OPT_LONGONLY("mime-type", 0, 0,
    "            output the MIME type\n", OPT_MIME_TYPE)
OPT_LONGONLY("mime-encoding", 0, 0,
    "        output the MIME encoding\n", OPT_MIME_ENCODING)
OPT('k', "keep-going", 0, 0,
    "           don't stop at the first match\n")
OPT('l', "list", 0, 0,
    "                 list magic strength\n")
#ifdef S_IFLNK
OPT('L', "dereference", 0, 1,
    "          follow symlinks")
OPT('h', "no-dereference", 0, 2,
    "       don't follow symlinks")
#endif
OPT('n', "no-buffer", 0, 0,
    "            do not buffer output\n")
OPT('N', "no-pad", 0, 0,
    "               do not pad output\n")
OPT('0', "print0", 0, 0,
    "               terminate filenames with ASCII NUL\n")
#if defined(HAVE_UTIME) || defined(HAVE_UTIMES)
OPT('p', "preserve-date", 0, 0,
    "        preserve access times on files\n")
#endif
OPT('P', "parameter", 1, 0,
    "            set file engine parameter limits\n"
    "                               %P\n")
OPT('r', "raw", 0, 0,
    "                  don't translate unprintable chars to \\ooo\n")
OPT('s', "special-files", 0, 0,
    "        treat special (block/char devices) files as\n"
    "                             ordinary ones\n")
OPT('S', "no-sandbox", 0, 0,
    "           disable system call sandboxing\n")
OPT('C', "compile", 0, 0,
    "              compile file specified by -m\n")
/* 'D' is always debug, 'd' is frequently something else */
OPT('D', "debug", 0, 0, "                print debugging messages\n")
