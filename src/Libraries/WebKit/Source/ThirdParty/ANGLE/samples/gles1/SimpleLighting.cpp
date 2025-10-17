/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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

class SimpleLightingSample : public SampleApplication
{
  public:
    SimpleLightingSample(int argc, char **argv)
        : SampleApplication("SimpleLightingSample", argc, argv, ClientType::ES1)
    {}

    bool initialize() override
    {
        glClearColor(0.4f, 0.3f, 0.2f, 1.0f);
        mRotDeg = 0.0f;

        return true;
    }

    void destroy() override {}

    void draw() override
    {

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor4f(0.2f, 0.6f, 0.8f, 1.0f);

        GLfloat mat_ambient[]  = {0.7f, 0.4f, 0.2f, 1.0f};
        GLfloat mat_specular[] = {0.5f, 0.5f, 0.5f, 1.0f};
        GLfloat mat_diffuse[]  = {0.3f, 0.4f, 0.6f, 1.0f};
        GLfloat lightpos[]     = {0.0f, 1.0f, 0.0f, 0.0f};

        GLfloat normals[] = {
            -0.4f, 0.4f, -0.4f, -0.4f, -0.4f, -0.4f, 0.2f, 0.0f, -0.4f,

            -0.4f, 0.4f, 0.4f,  -0.4f, -0.4f, 0.4f,  0.2f, 0.0f, 0.4f,
        };

        GLfloat vertices[] = {
            -0.5f, 0.5f, 0.0f, -0.5f, -0.5f, 0.0f, 0.5f, 0.0f, 0.0f,

            -0.5f, 0.5f, 0.3f, -0.5f, -0.5f, 0.3f, 0.5f, 0.0f, 0.3f,
        };

        GLuint indices[] = {
            0, 1, 2, 3, 4, 5,

            0, 4, 3, 4, 0, 1,

            4, 1, 2, 2, 5, 4,

            5, 2, 3, 3, 2, 0,
        };

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glLightfv(GL_LIGHT0, GL_POSITION, lightpos);

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glVertexPointer(3, GL_FLOAT, 0, vertices);
        glNormalPointer(GL_FLOAT, 0, normals);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                glPushMatrix();

                glTranslatef(-0.6f + i * 0.6f, -0.6f + j * 0.6f, 0.0f);

                glRotatef(mRotDeg + (10.0f * (3.0f * i + j)), 0.0f, 1.0f, 0.0f);
                glRotatef(20.0f + (20.0f * (3.0f * i + j)), 1.0f, 0.0f, 0.0f);
                GLfloat scale = 0.5;
                glScalef(scale, scale, scale);
                glDrawElements(GL_TRIANGLES, 3 * 8, GL_UNSIGNED_INT, indices);

                glPopMatrix();
            }
        }

        mRotDeg += 0.03f;
    }

  private:
    float mRotDeg;
};

int main(int argc, char **argv)
{
    SimpleLightingSample app(argc, argv);
    return app.run();
}
