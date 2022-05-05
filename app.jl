using Random
using GLMakie
using Base.Threads
using NearestNeighbors
using StaticArrays
using Profile
using ThreadsX
using TimerOutputs
using Distances
using OpenCL
#using ProfileView

# debug
const to = TimerOutput()


const kernelsource = "
   __kernel void vboids(__global const float *boids,
                     __global float *boids_o)
    {
        int offset = 6;
      int gid = get_global_id(0);
      for (int i = 0; i < 6; i++) {
        boids_o[i + (gid*offset) ] = boids[i + (gid*offset) ]*2.0;
      }
    }
"


n = 100*1000 # has to be divisible by threads
vmax = 0.03
d0 = 0.02
dc = 0.01
l0 = 0.31
l1 = 0.001
l2 = 1.2
l3 = 2
l4 = 0.01


locations(boids) = boids[1:3,:]
locations_t(boids) = transpose(boids[1:3,:])

# init
boids = 2.0f32*rand(Float32,(6,n)) .- 1.0f32
#boids[4:6,:] .= 0.0

kdtree = KDTree(locations(boids); leafsize = 10)

fig = Figure()
ax3d = Axis3(fig[1, 1]; aspect = (1, 1, 1),
    perspectiveness = 0.3, azimuth = 0.8, elevation = 0.30, show_axis = false)
#hidedecorations!(ax3d)
#hidespines!(ax3d)
points = Observable(locations_t(boids))
#velocities = Observable([Point3f(x.velocity[1], x.velocity[2], x.velocity[3]) for x in boids])
#arrows!(ax3d,points,velocities,linewidth = 0.01, arrowsize = Vec3f(0.2, 0.2, 0.3)*0.1, linecolor=:gray60, arrowcolor=:gray30)
scatter!(ax3d,points,markersize=10)
display(fig)

# multithreading
threads = 8

# pre alloc buffer
buffer = copy(boids)

# opencl
device, ctx, queue = cl.create_compute_context()
program = cl.Program(ctx, source=kernelsource) |> cl.build!

cl_b = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=boids[:])
cl_o = cl.Buffer(Float32, ctx, :w, n*6)

vboids = cl.Kernel(program, "vboids")

function update_sim() 
    @timeit to "process" begin
        Threads.@threads for t = 1:threads
            # calc positions
            offset = (n÷threads)*(t-1)
            slice = (1 + (n÷threads)*(t-1)):(t*(n÷threads))
            nb = first(knn(kdtree,boids[1:3,slice], 10, true))

            queue(vboids, n, nothing, cl_b, cl_o)

            #@inbounds for i = 1:size(nb)[1]
            #    index = i + offset
            #    #data = nb[i]
            #    #update_boid!(index, buffer, boids[:,index], boids, data)
            #end
        end
    end
    

    @timeit to "cleanup" begin
    global boids = copy(buffer)
    global kdtree = KDTree(locations(boids); leafsize = 10, reorder = false)
    end
end

(for t in 1:1000
    println(t)
    @timeit to "frame" update_sim()
    points[] = locations_t(boids)
    #velocities[] = [Point3f(x.velocity[1], x.velocity[2], x.velocity[3]) for x in boids]
    #sleep(0.0001)
end)

show(to)

"""record(fig, "append_animation.mp4", 1:30*20;
        framerate = 30) do frame
            println(frame)
            Threads.@threads for i = 1:length(buffer)
                buffer[i] = update_boid(buffer[i], boids, kdtree)
            end
            global boids = buffer
            points[] = [Point3f(x.position[1], x.position[2], x.position[3]) for x in boids]
            # updated indices
            global kdtree = KDTree(locations(boids); leafsize = 10)
        
            sleep(0.0001)
end"""