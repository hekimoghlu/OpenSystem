/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_fd_stream.h"

#include "pas_log.h"

static PAS_FORMAT_PRINTF(2, 0) void fd_stream_vprintf(pas_stream* stream, const char* format, va_list arg_list)
{
    pas_fd_stream_vprintf((pas_fd_stream*)stream, format, arg_list);
}

pas_stream_functions pas_fd_stream_functions = {
    .vprintf = fd_stream_vprintf
};

pas_fd_stream pas_log_stream = PAS_FD_STREAM_INITIALIZER(PAS_LOG_DEFAULT_FD);

void pas_fd_stream_construct(pas_fd_stream* stream, int fd)
{
    stream->base.functions = &pas_fd_stream_functions;
    stream->fd = fd;
}

void pas_fd_stream_vprintf(pas_fd_stream* stream, const char* format, va_list arg_list)
{
    pas_vlog_fd(stream->fd, format, arg_list);
}

#endif /* LIBPAS_ENABLED */
