/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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
#include "curlcheck.h"

#include "curl_fnmatch.h"

static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{
}

#ifndef CURL_DISABLE_FTP

/*
   CURL_FNMATCH_MATCH    0
   CURL_FNMATCH_NOMATCH  1
   CURL_FNMATCH_FAIL     2
 */

#define MATCH   CURL_FNMATCH_MATCH
#define NOMATCH CURL_FNMATCH_NOMATCH

#define LINUX_DIFFER 0x80
#define LINUX_SHIFT 8
#define LINUX_MATCH ((CURL_FNMATCH_MATCH << LINUX_SHIFT) | LINUX_DIFFER)
#define LINUX_NOMATCH ((CURL_FNMATCH_NOMATCH << LINUX_SHIFT) | LINUX_DIFFER)
#define LINUX_FAIL ((CURL_FNMATCH_FAIL << LINUX_SHIFT) | LINUX_DIFFER)

#define MAC_DIFFER 0x40
#define MAC_SHIFT 16
#define MAC_MATCH ((CURL_FNMATCH_MATCH << MAC_SHIFT) | MAC_DIFFER)
#define MAC_NOMATCH ((CURL_FNMATCH_NOMATCH << MAC_SHIFT) | MAC_DIFFER)
#define MAC_FAIL ((CURL_FNMATCH_FAIL << MAC_SHIFT) | MAC_DIFFER)

struct testcase {
  const char *pattern;
  const char *string;
  int result;
};

static const struct testcase tests[] = {
  /* brackets syntax */
  {"*[*[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
   "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
   "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[\001\177[[[[[[[[[[[[[[[[[[[[[",
   "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
   "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
   "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[",
   NOMATCH|MAC_FAIL},

  { "\\[",                      "[",                      MATCH },
  { "[",                        "[",             NOMATCH|LINUX_MATCH|MAC_FAIL},
  { "[]",                       "[]",            NOMATCH|LINUX_MATCH|MAC_FAIL},
  { "[][]",                     "[",                      MATCH },
  { "[][]",                     "]",                      MATCH },
  { "[[]",                      "[",                      MATCH },
  { "[[[]",                     "[",                      MATCH },
  { "[[[[]",                    "[",                      MATCH },
  { "[[[[]",                    "[",                      MATCH },

  { "[][[]",                    "]",                      MATCH },
  { "[][[[]",                   "[",                      MATCH },
  { "[[]",                      "]",                      NOMATCH },

  { "[a@]",                     "a",                      MATCH },

  { "[a-z]",                    "a",                      MATCH },
  { "[a-z]",                    "A",                      NOMATCH },
  { "?[a-z]",                   "?Z",                     NOMATCH },
  { "[A-Z]",                    "C",                      MATCH },
  { "[A-Z]",                    "c",                      NOMATCH },
  { "[0-9]",                    "7",                      MATCH },
  { "[7-8]",                    "7",                      MATCH },
  { "[7-]",                     "7",                      MATCH },
  { "[7-]",                     "-",                      MATCH },
  { "[7-]",                     "[",                      NOMATCH },
  { "[a-bA-F]",                 "F",                      MATCH },
  { "[a-bA-B9]",                "9",                      MATCH },
  { "[a-bA-B98]",               "8",                      MATCH },
  { "[a-bA-B98]",               "C",                      NOMATCH },
  { "[a-bA-Z9]",                "F",                      MATCH },
  { "[a-bA-Z9]ero*",            "Zero chance.",           MATCH },
  { "S[a-][x]opho*",            "Saxophone",              MATCH },
  { "S[a-][x]opho*",            "SaXophone",              NOMATCH },
  { "S[a-][x]*.txt",            "S-x.txt",                MATCH },
  { "[\\a-\\b]",                "a",                      MATCH },
  { "[\\a-\\b]",                "b",                      MATCH },
  { "[?*[][?*[][?*[]",          "?*[",                    MATCH },
  { "[][?*-]",                  "]",                      MATCH },
  { "[][?*-]",                  "[",                      MATCH },
  { "[][?*-]",                  "?",                      MATCH },
  { "[][?*-]",                  "*",                      MATCH },
  { "[][?*-]",                  "-",                      MATCH },
  { "[]?*-]",                   "-",                      MATCH },
  { "[\xFF]",                   "\xFF", MATCH|LINUX_FAIL|MAC_FAIL},
  { "?/b/c",                    "a/b/c",                  MATCH },
  { "^_{}~",                    "^_{}~",                  MATCH },
  { "!#%+,-./01234567889",      "!#%+,-./01234567889",    MATCH },
  { "PQRSTUVWXYZ]abcdefg",      "PQRSTUVWXYZ]abcdefg",    MATCH },
  { ":;=@ABCDEFGHIJKLMNO",      ":;=@ABCDEFGHIJKLMNO",    MATCH },

  /* negate */
  { "[!a]",                     "b",                      MATCH },
  { "[!a]",                     "a",                      NOMATCH },
  { "[^a]",                     "b",                      MATCH },
  { "[^a]",                     "a",                      NOMATCH },
  { "[^a-z0-9A-Z]",             "a",                      NOMATCH },
  { "[^a-z0-9A-Z]",             "-",                      MATCH },
  { "curl[!a-z]lib",            "curl lib",               MATCH },
  { "curl[! ]lib",              "curl lib",               NOMATCH },
  { "[! ][ ]",                  "  ",                     NOMATCH },
  { "[! ][ ]",                  "a ",                     MATCH },
  { "*[^a].t?t",                "a.txt",                  NOMATCH },
  { "*[^a].t?t",                "ba.txt",                 NOMATCH },
  { "*[^a].t?t",                "ab.txt",                 MATCH },
  { "*[^a]",                    "",                       NOMATCH },
  { "[!\xFF]",                  "",             NOMATCH|LINUX_FAIL},
  { "[!\xFF]",                  "\xFF",  NOMATCH|LINUX_FAIL|MAC_FAIL},
  { "[!\xFF]",                  "a",      MATCH|LINUX_FAIL|MAC_FAIL},
  { "[!?*[]",                   "?",                      NOMATCH },
  { "[!!]",                     "!",                      NOMATCH },
  { "[!!]",                     "x",                      MATCH },

  { "[[:alpha:]]",              "a",                      MATCH },
  { "[[:alpha:]]",              "9",                      NOMATCH },
  { "[[:alnum:]]",              "a",                      MATCH },
  { "[[:alnum:]]",              "[",                      NOMATCH },
  { "[[:alnum:]]",              "]",                      NOMATCH },
  { "[[:alnum:]]",              "9",                      MATCH },
  { "[[:digit:]]",              "9",                      MATCH },
  { "[[:xdigit:]]",             "9",                      MATCH },
  { "[[:xdigit:]]",             "F",                      MATCH },
  { "[[:xdigit:]]",             "G",                      NOMATCH },
  { "[[:upper:]]",              "U",                      MATCH },
  { "[[:upper:]]",              "u",                      NOMATCH },
  { "[[:lower:]]",              "l",                      MATCH },
  { "[[:lower:]]",              "L",                      NOMATCH },
  { "[[:print:]]",              "L",                      MATCH },
  { "[[:print:]]",              "\10",                    NOMATCH },
  { "[[:print:]]",              "\10",                    NOMATCH },
  { "[[:space:]]",              " ",                      MATCH },
  { "[[:space:]]",              "x",                      NOMATCH },
  { "[[:graph:]]",              " ",                      NOMATCH },
  { "[[:graph:]]",              "x",                      MATCH },
  { "[[:blank:]]",              "\t",                     MATCH },
  { "[[:blank:]]",              " ",                      MATCH },
  { "[[:blank:]]",              "\r",                     NOMATCH },
  { "[^[:blank:]]",             "\t",                     NOMATCH },
  { "[^[:print:]]",             "\10",                    MATCH },
  { "[[:lower:]][[:lower:]]",   "ll",                     MATCH },
  { "[[:foo:]]",                "bar",                    NOMATCH|MAC_FAIL},
  { "[[:foo:]]",                "f]",         MATCH|LINUX_NOMATCH|MAC_FAIL},

  { "Curl[[:blank:]];-)",       "Curl ;-)",               MATCH },
  { "*[[:blank:]]*",            " ",                      MATCH },
  { "*[[:blank:]]*",            "",                       NOMATCH },
  { "*[[:blank:]]*",            "hi, im_Pavel",           MATCH },

  /* common using */
  { "filename.dat",             "filename.dat",           MATCH },
  { "*curl*",                   "lets use curl!!",        MATCH },
  { "filename.txt",             "filename.dat",           NOMATCH },
  { "*.txt",                    "text.txt",               MATCH },
  { "*.txt",                    "a.txt",                  MATCH },
  { "*.txt",                    ".txt",                   MATCH },
  { "*.txt",                    "txt",                    NOMATCH },
  { "??.txt",                   "99.txt",                 MATCH },
  { "??.txt",                   "a99.txt",                NOMATCH },
  { "?.???",                    "a.txt",                  MATCH },
  { "*.???",                    "somefile.dat",           MATCH },
  { "*.???",                    "photo.jpeg",             NOMATCH },
  { ".*",                       ".htaccess",              MATCH },
  { ".*",                       ".",                      MATCH },
  { ".*",                       "..",                     MATCH },

  /* many stars => one star */
  { "**.txt",                   "text.txt",               MATCH },
  { "***.txt",                  "t.txt",                  MATCH },
  { "****.txt",                 ".txt",                   MATCH },

  /* empty string or pattern */
  { "",                         "",                       MATCH },
  { "",                         "hello",                  NOMATCH },
  { "file",                     "",                       NOMATCH  },
  { "?",                        "",                       NOMATCH },
  { "*",                        "",                       MATCH },
  { "x",                        "",                       NOMATCH },

  /* backslash */
  { "\\",                       "\\",                     MATCH|LINUX_NOMATCH},
  { "\\\\",                     "\\",                     MATCH },
  { "\\\\",                     "\\\\",                   NOMATCH },
  { "\\?",                      "?",                      MATCH },
  { "\\*",                      "*",                      MATCH },
  { "?.txt",                    "?.txt",                  MATCH },
  { "*.txt",                    "*.txt",                  MATCH },
  { "\\?.txt",                  "?.txt",                  MATCH },
  { "\\*.txt",                  "*.txt",                  MATCH },
  { "\\?.txt",                  "x.txt",                  NOMATCH },
  { "\\*.txt",                  "x.txt",                  NOMATCH },
  { "\\*\\\\.txt",              "*\\.txt",                MATCH },
  { "*\\**\\?*\\\\*",           "cc*cc?cccc",             NOMATCH },
  { "*\\?*\\**",                "cc?cc",                  NOMATCH },
  { "\\\"\\$\\&\\'\\(\\)",      "\"$&'()",                MATCH },
  { "\\*\\?\\[\\\\\\`\\|",      "*?[\\`|",                MATCH },
  { "[\\a\\b]c",                "ac",                     MATCH },
  { "[\\a\\b]c",                "bc",                     MATCH },
  { "[\\a\\b]d",                "bc",                     NOMATCH },
  { "[a-bA-B\\?]",              "?",                      MATCH },
  { "cu[a-ab-b\\r]l",           "curl",                   MATCH },
  { "[\\a-z]",                  "c",                      MATCH },

  { "?*?*?.*?*",                "abc.c",                  MATCH },
  { "?*?*?.*?*",                "abcc",                   NOMATCH },
  { "?*?*?.*?*",                "abc.",                   NOMATCH },
  { "?*?*?.*?*",                "abc.c++",                MATCH },
  { "?*?*?.*?*",                "abcdef.c++",             MATCH },
  { "?*?*?.?",                  "abcdef.c",               MATCH },
  { "?*?*?.?",                  "abcdef.cd",              NOMATCH },

  { "LindmÃ¤tarv",               "LindmÃ¤tarv",             MATCH },

  { "",                         "",                       MATCH},
  {"**]*[*[\x13]**[*\x13)]*]*[**[*\x13~r-]*]**[.*]*[\xe3\xe3\xe3\xe3\xe3\xe3"
   "\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3"
   "\xe3\xe3\xe3\xe3\xe3*[\x13]**[*\x13)]*]*[*[\x13]*[~r]*]*\xba\x13\xa6~b-]*",
                                "a",                      NOMATCH|LINUX_FAIL}
};

static const char *ret2name(int i)
{
  switch(i) {
  case 0:
    return "MATCH";
  case 1:
    return "NOMATCH";
  case 2:
    return "FAIL";
  default:
    return "unknown";
  }
  /* not reached */
}

enum system {
  SYSTEM_CUSTOM,
  SYSTEM_LINUX,
  SYSTEM_MACOS
};

UNITTEST_START
{
  int testnum = sizeof(tests) / sizeof(struct testcase);
  int i;
  enum system machine;

#ifdef HAVE_FNMATCH
  if(strstr(OS, "apple") || strstr(OS, "darwin")) {
    machine = SYSTEM_MACOS;
  }
  else
    machine = SYSTEM_LINUX;
  printf("Tested with system fnmatch(), %s-style\n",
         machine == SYSTEM_LINUX ? "linux" : "mac");
#else
  printf("Tested with custom fnmatch()\n");
  machine = SYSTEM_CUSTOM;
#endif

  for(i = 0; i < testnum; i++) {
    int result = tests[i].result;
    int rc = Curl_fnmatch(NULL, tests[i].pattern, tests[i].string);
    if(result & (LINUX_DIFFER|MAC_DIFFER)) {
      if((result & LINUX_DIFFER) && (machine == SYSTEM_LINUX))
        result >>= LINUX_SHIFT;
      else if((result & MAC_DIFFER) && (machine == SYSTEM_MACOS))
        result >>= MAC_SHIFT;
      result &= 0x03; /* filter off all high bits */
    }
    if(rc != result) {
      printf("Curl_fnmatch(\"%s\", \"%s\") should return %s (returns %s)"
             " [%d]\n",
             tests[i].pattern, tests[i].string, ret2name(result),
             ret2name(rc), i);
      fail("pattern mismatch");
    }
  }
}
UNITTEST_STOP

#else

UNITTEST_START
UNITTEST_STOP

#endif
