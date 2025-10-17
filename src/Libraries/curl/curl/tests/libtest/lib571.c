/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#include "test.h"

#ifdef HAVE_NETINET_IN_H
#  include <netinet/in.h>
#endif
#ifdef HAVE_NETDB_H
#  include <netdb.h>
#endif
#ifdef HAVE_ARPA_INET_H
#  include <arpa/inet.h>
#endif
#ifdef HAVE_SYS_STAT_H
#  include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
#  include <fcntl.h>
#endif

#include "warnless.h"
#include "memdebug.h"

#define RTP_PKT_CHANNEL(p)   ((int)((unsigned char)((p)[1])))

#define RTP_PKT_LENGTH(p)  ((((int)((unsigned char)((p)[2]))) << 8) | \
                             ((int)((unsigned char)((p)[3]))))

#define RTP_DATA_SIZE 12
static const char *RTP_DATA = "$_1234\n\0Rsdf";

static int rtp_packet_count = 0;

static size_t rtp_write(void *ptr, size_t size, size_t nmemb, void *stream)
{
  char *data = (char *)ptr;
  int channel = RTP_PKT_CHANNEL(data);
  int message_size;
  int coded_size = RTP_PKT_LENGTH(data);
  size_t failure = (size && nmemb) ? 0 : 1;
  int i;
  (void)stream;

  message_size = curlx_uztosi(size * nmemb) - 4;

  printf("RTP: message size %d, channel %d\n", message_size, channel);
  if(message_size != coded_size) {
    printf("RTP embedded size (%d) does not match the write size (%d).\n",
           coded_size, message_size);
    return failure;
  }

  data += 4;
  for(i = 0; i < message_size; i += RTP_DATA_SIZE) {
    if(message_size - i > RTP_DATA_SIZE) {
      if(memcmp(RTP_DATA, data + i, RTP_DATA_SIZE) != 0) {
        printf("RTP PAYLOAD CORRUPTED [%s]\n", data + i);
        /* return failure; */
      }
    }
    else {
      if(memcmp(RTP_DATA, data + i, message_size - i) != 0) {
        printf("RTP PAYLOAD END CORRUPTED (%d), [%s]\n",
               message_size - i, data + i);
        /* return failure; */
      }
    }
  }

  rtp_packet_count++;
  fprintf(stderr, "packet count is %d\n", rtp_packet_count);

  return size * nmemb;
}

/* build request url */
static char *suburl(const char *base, int i)
{
  return curl_maprintf("%s%.4d", base, i);
}

int test(char *URL)
{
  int res;
  CURL *curl;
  char *stream_uri = NULL;
  int request = 1;

  FILE *protofile = fopen(libtest_arg2, "wb");
  if(!protofile) {
    fprintf(stderr, "Couldn't open the protocol dump file\n");
    return TEST_ERR_MAJOR_BAD;
  }

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    fclose(protofile);
    return TEST_ERR_MAJOR_BAD;
  }

  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    fclose(protofile);
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }
  test_setopt(curl, CURLOPT_URL, URL);

  stream_uri = suburl(URL, request++);
  if(!stream_uri) {
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }
  test_setopt(curl, CURLOPT_RTSP_STREAM_URI, stream_uri);
  curl_free(stream_uri);
  stream_uri = NULL;

  test_setopt(curl, CURLOPT_INTERLEAVEFUNCTION, rtp_write);
  test_setopt(curl, CURLOPT_TIMEOUT, 30L);
  test_setopt(curl, CURLOPT_VERBOSE, 1L);
  test_setopt(curl, CURLOPT_WRITEDATA, protofile);

  test_setopt(curl, CURLOPT_RTSP_TRANSPORT, "RTP/AVP/TCP;interleaved=0-1");
  test_setopt(curl, CURLOPT_RTSP_REQUEST, CURL_RTSPREQ_SETUP);

  res = curl_easy_perform(curl);
  if(res)
    goto test_cleanup;

  /* This PLAY starts the interleave */
  stream_uri = suburl(URL, request++);
  if(!stream_uri) {
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }
  test_setopt(curl, CURLOPT_RTSP_STREAM_URI, stream_uri);
  curl_free(stream_uri);
  stream_uri = NULL;
  test_setopt(curl, CURLOPT_RTSP_REQUEST, CURL_RTSPREQ_PLAY);

  res = curl_easy_perform(curl);
  if(res)
    goto test_cleanup;

  /* The DESCRIBE request will try to consume data after the Content */
  stream_uri = suburl(URL, request++);
  if(!stream_uri) {
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }
  test_setopt(curl, CURLOPT_RTSP_STREAM_URI, stream_uri);
  curl_free(stream_uri);
  stream_uri = NULL;
  test_setopt(curl, CURLOPT_RTSP_REQUEST, CURL_RTSPREQ_DESCRIBE);

  res = curl_easy_perform(curl);
  if(res)
    goto test_cleanup;

  stream_uri = suburl(URL, request++);
  if(!stream_uri) {
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }
  test_setopt(curl, CURLOPT_RTSP_STREAM_URI, stream_uri);
  curl_free(stream_uri);
  stream_uri = NULL;
  test_setopt(curl, CURLOPT_RTSP_REQUEST, CURL_RTSPREQ_PLAY);

  res = curl_easy_perform(curl);
  if(res)
    goto test_cleanup;

  fprintf(stderr, "PLAY COMPLETE\n");

  /* Use Receive to get the rest of the data */
  while(!res && rtp_packet_count < 19) {
    fprintf(stderr, "LOOPY LOOP!\n");
    test_setopt(curl, CURLOPT_RTSP_REQUEST, CURL_RTSPREQ_RECEIVE);
    res = curl_easy_perform(curl);
  }

test_cleanup:
  curl_free(stream_uri);

  if(protofile)
    fclose(protofile);

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res;
}
