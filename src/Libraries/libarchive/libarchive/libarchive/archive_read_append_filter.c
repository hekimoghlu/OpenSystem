/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#include "archive_platform.h"

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#include "archive.h"
#include "archive_private.h"
#include "archive_read_private.h"

int
archive_read_append_filter(struct archive *_a, int code)
{
  int r1, r2, number_bidders, i;
  char str[20];
  struct archive_read_filter_bidder *bidder;
  struct archive_read_filter *filter;
  struct archive_read *a = (struct archive_read *)_a;

  r2 = (ARCHIVE_OK);
  switch (code)
  {
    case ARCHIVE_FILTER_NONE:
      /* No filter to add, so do nothing.
       * NOTE: An initial "NONE" type filter is always set at the end of the
       * filter chain.
       */
      r1 = (ARCHIVE_OK);
      break;
    case ARCHIVE_FILTER_GZIP:
      strcpy(str, "gzip");
      r1 = archive_read_support_filter_gzip(_a);
      break;
    case ARCHIVE_FILTER_BZIP2:
      strcpy(str, "bzip2");
      r1 = archive_read_support_filter_bzip2(_a);
      break;
    case ARCHIVE_FILTER_COMPRESS:
      strcpy(str, "compress (.Z)");
      r1 = archive_read_support_filter_compress(_a);
      break;
    case ARCHIVE_FILTER_PROGRAM:
      archive_set_error(&a->archive, ARCHIVE_ERRNO_PROGRAMMER,
          "Cannot append program filter using archive_read_append_filter");
      return (ARCHIVE_FATAL);
    case ARCHIVE_FILTER_LZMA:
      strcpy(str, "lzma");
      r1 = archive_read_support_filter_lzma(_a);
      break;
    case ARCHIVE_FILTER_XZ:
      strcpy(str, "xz");
      r1 = archive_read_support_filter_xz(_a);
      break;
    case ARCHIVE_FILTER_UU:
      strcpy(str, "uu");
      r1 = archive_read_support_filter_uu(_a);
      break;
    case ARCHIVE_FILTER_RPM:
      strcpy(str, "rpm");
      r1 = archive_read_support_filter_rpm(_a);
      break;
    case ARCHIVE_FILTER_LZ4:
      strcpy(str, "lz4");
      r1 = archive_read_support_filter_lz4(_a);
      break;
    case ARCHIVE_FILTER_ZSTD:
      strcpy(str, "zstd");
      r1 = archive_read_support_filter_zstd(_a);
      break;
    case ARCHIVE_FILTER_LZIP:
      strcpy(str, "lzip");
      r1 = archive_read_support_filter_lzip(_a);
      break;
    case ARCHIVE_FILTER_LRZIP:
      strcpy(str, "lrzip");
      r1 = archive_read_support_filter_lrzip(_a);
      break;
    default:
      archive_set_error(&a->archive, ARCHIVE_ERRNO_PROGRAMMER,
          "Invalid filter code specified");
      return (ARCHIVE_FATAL);
  }

  if (code != ARCHIVE_FILTER_NONE)
  {
    number_bidders = sizeof(a->bidders) / sizeof(a->bidders[0]);

    bidder = a->bidders;
    for (i = 0; i < number_bidders; i++, bidder++)
    {
      if (!bidder->name || !strcmp(bidder->name, str))
        break;
    }
    if (!bidder->name || strcmp(bidder->name, str))
    {
      archive_set_error(&a->archive, ARCHIVE_ERRNO_PROGRAMMER,
          "Internal error: Unable to append filter");
      return (ARCHIVE_FATAL);
    }

    filter
        = (struct archive_read_filter *)calloc(1, sizeof(*filter));
    if (filter == NULL)
    {
      archive_set_error(&a->archive, ENOMEM, "Out of memory");
      return (ARCHIVE_FATAL);
    }
    filter->bidder = bidder;
    filter->archive = a;
    filter->upstream = a->filter;
    a->filter = filter;
    r2 = (bidder->vtable->init)(a->filter);
    if (r2 != ARCHIVE_OK) {
      __archive_read_free_filters(a);
      return (ARCHIVE_FATAL);
    }
  }

  a->bypass_filter_bidding = 1;
  return (r1 < r2) ? r1 : r2;
}

int
archive_read_append_filter_program(struct archive *_a, const char *cmd)
{
  return (archive_read_append_filter_program_signature(_a, cmd, NULL, 0));
}

int
archive_read_append_filter_program_signature(struct archive *_a,
  const char *cmd, const void *signature, size_t signature_len)
{
  int r, number_bidders, i;
  struct archive_read_filter_bidder *bidder;
  struct archive_read_filter *filter;
  struct archive_read *a = (struct archive_read *)_a;

  if (archive_read_support_filter_program_signature(_a, cmd, signature,
    signature_len) != (ARCHIVE_OK))
    return (ARCHIVE_FATAL);

  number_bidders = sizeof(a->bidders) / sizeof(a->bidders[0]);

  bidder = a->bidders;
  for (i = 0; i < number_bidders; i++, bidder++)
  {
    /* Program bidder name set to filter name after initialization */
    if (bidder->data && !bidder->name)
      break;
  }
  if (!bidder->data)
  {
    archive_set_error(&a->archive, ARCHIVE_ERRNO_PROGRAMMER,
        "Internal error: Unable to append program filter");
    return (ARCHIVE_FATAL);
  }

  filter
      = (struct archive_read_filter *)calloc(1, sizeof(*filter));
  if (filter == NULL)
  {
    archive_set_error(&a->archive, ENOMEM, "Out of memory");
    return (ARCHIVE_FATAL);
  }
  filter->bidder = bidder;
  filter->archive = a;
  filter->upstream = a->filter;
  a->filter = filter;
  r = (bidder->vtable->init)(a->filter);
  if (r != ARCHIVE_OK) {
    __archive_read_free_filters(a);
    return (ARCHIVE_FATAL);
  }
  bidder->name = a->filter->name;

  a->bypass_filter_bidding = 1;
  return r;
}
