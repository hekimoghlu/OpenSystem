/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
#include <inttypes.h>
#include <string.h>

#include "dav1d/dav1d.h"

#include <SDL.h>
#ifdef HAVE_PLACEBO
# include <libplacebo/config.h>
#endif

// Check libplacebo Vulkan rendering
#if defined(HAVE_VULKAN) && defined(SDL_VIDEO_VULKAN)
# if defined(PL_HAVE_VULKAN) && PL_HAVE_VULKAN
#  define HAVE_RENDERER_PLACEBO
#  define HAVE_PLACEBO_VULKAN
# endif
#endif

// Check libplacebo OpenGL rendering
#if defined(PL_HAVE_OPENGL) && PL_HAVE_OPENGL
# define HAVE_RENDERER_PLACEBO
# define HAVE_PLACEBO_OPENGL
#endif

/**
 * Settings structure
 * Hold all settings available for the player,
 * this is usually filled by parsing arguments
 * from the console.
 */
typedef struct {
    const char *inputfile;
    const char *renderer_name;
    int highquality;
    int untimed;
    int zerocopy;
    int gpugrain;
} Dav1dPlaySettings;

#define WINDOW_WIDTH  910
#define WINDOW_HEIGHT 512

enum {
    DAV1D_EVENT_NEW_FRAME,
    DAV1D_EVENT_SEEK_FRAME,
    DAV1D_EVENT_DEC_QUIT
};

/**
 * Renderer info
 */
typedef struct rdr_info
{
    // Renderer name
    const char *name;
    // Cookie passed to the renderer implementation callbacks
    void *cookie;
    // Callback to create the renderer
    void* (*create_renderer)(void);
    // Callback to destroy the renderer
    void (*destroy_renderer)(void *cookie);
    // Callback to the render function that renders a prevously sent frame
    void (*render)(void *cookie, const Dav1dPlaySettings *settings);
    // Callback to the send frame function, _may_ also unref dav1d_pic!
    int (*update_frame)(void *cookie, Dav1dPicture *dav1d_pic,
                        const Dav1dPlaySettings *settings);
    // Callback for alloc/release pictures (optional)
    int (*alloc_pic)(Dav1dPicture *pic, void *cookie);
    void (*release_pic)(Dav1dPicture *pic, void *cookie);
    // Whether or not this renderer can apply on-GPU film grain synthesis
    int supports_gpu_grain;
} Dav1dPlayRenderInfo;

extern const Dav1dPlayRenderInfo rdr_placebo_vk;
extern const Dav1dPlayRenderInfo rdr_placebo_gl;
extern const Dav1dPlayRenderInfo rdr_sdl;

// Available renderes ordered by priority
static const Dav1dPlayRenderInfo* const dp_renderers[] = {
    &rdr_placebo_vk,
    &rdr_placebo_gl,
    &rdr_sdl,
};

static inline const Dav1dPlayRenderInfo *dp_get_renderer(const char *name)
{
    for (size_t i = 0; i < (sizeof(dp_renderers)/sizeof(*dp_renderers)); ++i)
    {
        if (dp_renderers[i]->name == NULL)
            continue;

        if (name == NULL || strcmp(name, dp_renderers[i]->name) == 0) {
            return dp_renderers[i];
        }
    }
    return NULL;
}

static inline SDL_Window *dp_create_sdl_window(int window_flags)
{
    SDL_Window *win;
    window_flags |= SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI;

    win = SDL_CreateWindow("Dav1dPlay", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT, window_flags);
    SDL_SetWindowResizable(win, SDL_TRUE);

    return win;
}
