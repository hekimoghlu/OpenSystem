/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 27, 2022.
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

//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

//            Based on Hello_Triangle.c from
// Book:      OpenGL(R) ES 2.0 Programming Guide
// Authors:   Aaftab Munshi, Dan Ginsburg, Dave Shreiner
// ISBN-10:   0321502795
// ISBN-13:   9780321502797
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780321563835
//            http://www.opengles-book.com

#include "SampleApplication.h"

#include <algorithm>

#include "util/gles_loader_autogen.h"

class FlatShadingSample : public SampleApplication
{
  public:
    FlatShadingSample(int argc, char **argv)
        : SampleApplication("FlatShadingSample", argc, argv, ClientType::ES1)
    {}

    bool initialize() override
    {
        glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
        mRotDeg = 0.0f;

        return true;
    }

    void draw() override
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        GLfloat vertices[] = {
            -0.5f, 0.5f, 0.0f, -0.5f, -0.5f, 0.0f, 0.5f,  -0.5f,
            0.0f,  0.5f, 0.5f, 0.0f,  0.0f,  0.0f, -1.0f,
        };

        GLfloat colors[] = {
            1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        };

        GLuint indices[] = {
            0, 1, 2, 2, 3, 0,

            4, 1, 0, 4, 2, 1, 4, 3, 2, 4, 0, 3,
        };

        glEnable(GL_DEPTH_TEST);

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glVertexPointer(3, GL_FLOAT, 0, vertices);
        glColorPointer(4, GL_FLOAT, 0, colors);

        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                if ((i + j * 6) % 2 == 0)
                {
                    glShadeModel(GL_FLAT);
                }
                else
                {
                    glShadeModel(GL_SMOOTH);
                }

                glPushMatrix();

                glTranslatef(-0.7f + i * 0.3f, -0.7f + j * 0.3f, 0.0f);

                glRotatef(mRotDeg + (5.0f * (6.0f * i + j)), 0.0f, 1.0f, 0.0f);
                glRotatef(20.0f + (10.0f * (6.0f * i + j)), 1.0f, 0.0f, 0.0f);
                GLfloat scale = 0.2f;
                glScalef(scale, scale, scale);
                glDrawElements(GL_TRIANGLES, sizeof(indices) / sizeof(GLuint), GL_UNSIGNED_INT,
                               indices);

                glPopMatrix();
            }
        }

        mRotDeg += 0.1f;
    }

  private:
    float mRotDeg = 0.0f;
};

int main(int argc, char **argv)
{
    FlatShadingSample app(argc, argv);
    return app.run();
}
