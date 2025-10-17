/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
#include "netrc.h"
#include "memdebug.h" /* LAST include file */

#ifndef CURL_DISABLE_NETRC

static char *login;
static char *password;

static CURLcode unit_setup(void)
{
  password = strdup("");
  login = strdup("");
  if(!password || !login) {
    Curl_safefree(password);
    Curl_safefree(login);
    return CURLE_OUT_OF_MEMORY;
  }
  return CURLE_OK;
}

static void unit_stop(void)
{
  Curl_safefree(password);
  Curl_safefree(login);
}

UNITTEST_START
  int result;

  /*
   * Test a non existent host in our netrc file.
   */
  result = Curl_parsenetrc("test.example.com", &login, &password, arg);
  fail_unless(result == 1, "Host not found should return 1");
  abort_unless(password != NULL, "returned NULL!");
  fail_unless(password[0] == 0, "password should not have been changed");
  abort_unless(login != NULL, "returned NULL!");
  fail_unless(login[0] == 0, "login should not have been changed");

  /*
   * Test a non existent login in our netrc file.
   */
  free(login);
  login = strdup("me");
  abort_unless(login != NULL, "returned NULL!");
  result = Curl_parsenetrc("example.com", &login, &password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(password != NULL, "returned NULL!");
  fail_unless(password[0] == 0, "password should not have been changed");
  abort_unless(login != NULL, "returned NULL!");
  fail_unless(strncmp(login, "me", 2) == 0,
              "login should not have been changed");

  /*
   * Test a non existent login and host in our netrc file.
   */
  free(login);
  login = strdup("me");
  abort_unless(login != NULL, "returned NULL!");
  result = Curl_parsenetrc("test.example.com", &login, &password, arg);
  fail_unless(result == 1, "Host not found should return 1");
  abort_unless(password != NULL, "returned NULL!");
  fail_unless(password[0] == 0, "password should not have been changed");
  abort_unless(login != NULL, "returned NULL!");
  fail_unless(strncmp(login, "me", 2) == 0,
              "login should not have been changed");

  /*
   * Test a non existent login (substring of an existing one) in our
   * netrc file.
   */
  free(login);
  login = strdup("admi");
  abort_unless(login != NULL, "returned NULL!");
  result = Curl_parsenetrc("example.com", &login, &password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(password != NULL, "returned NULL!");
  fail_unless(password[0] == 0, "password should not have been changed");
  abort_unless(login != NULL, "returned NULL!");
  fail_unless(strncmp(login, "admi", 4) == 0,
              "login should not have been changed");

  /*
   * Test a non existent login (superstring of an existing one)
   * in our netrc file.
   */
  free(login);
  login = strdup("adminn");
  abort_unless(login != NULL, "returned NULL!");
  result = Curl_parsenetrc("example.com", &login, &password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(password != NULL, "returned NULL!");
  fail_unless(password[0] == 0, "password should not have been changed");
  abort_unless(login != NULL, "returned NULL!");
  fail_unless(strncmp(login, "adminn", 6) == 0,
              "login should not have been changed");

  /*
   * Test for the first existing host in our netrc file
   * with login[0] = 0.
   */
  free(login);
  login = strdup("");
  abort_unless(login != NULL, "returned NULL!");
  result = Curl_parsenetrc("example.com", &login, &password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(password != NULL, "returned NULL!");
  fail_unless(strncmp(password, "passwd", 6) == 0,
              "password should be 'passwd'");
  abort_unless(login != NULL, "returned NULL!");
  fail_unless(strncmp(login, "admin", 5) == 0, "login should be 'admin'");

  /*
   * Test for the first existing host in our netrc file
   * with login[0] != 0.
   */
  free(password);
  password = strdup("");
  abort_unless(password != NULL, "returned NULL!");
  result = Curl_parsenetrc("example.com", &login, &password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(password != NULL, "returned NULL!");
  fail_unless(strncmp(password, "passwd", 6) == 0,
              "password should be 'passwd'");
  abort_unless(login != NULL, "returned NULL!");
  fail_unless(strncmp(login, "admin", 5) == 0, "login should be 'admin'");

  /*
   * Test for the second existing host in our netrc file
   * with login[0] = 0.
   */
  free(password);
  password = strdup("");
  abort_unless(password != NULL, "returned NULL!");
  free(login);
  login = strdup("");
  abort_unless(login != NULL, "returned NULL!");
  result = Curl_parsenetrc("curl.example.com", &login, &password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(password != NULL, "returned NULL!");
  fail_unless(strncmp(password, "none", 4) == 0,
              "password should be 'none'");
  abort_unless(login != NULL, "returned NULL!");
  fail_unless(strncmp(login, "none", 4) == 0, "login should be 'none'");

  /*
   * Test for the second existing host in our netrc file
   * with login[0] != 0.
   */
  free(password);
  password = strdup("");
  abort_unless(password != NULL, "returned NULL!");
  result = Curl_parsenetrc("curl.example.com", &login, &password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(password != NULL, "returned NULL!");
  fail_unless(strncmp(password, "none", 4) == 0,
              "password should be 'none'");
  abort_unless(login != NULL, "returned NULL!");
  fail_unless(strncmp(login, "none", 4) == 0, "login should be 'none'");

UNITTEST_STOP

#else
static CURLcode unit_setup(void)
{
  return CURLE_OK;
}
static void unit_stop(void)
{
}
UNITTEST_START
UNITTEST_STOP

#endif
