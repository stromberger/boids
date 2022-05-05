using Random
using GLMakie
using Base.Threads
using NearestNeighbors
using StaticArrays
using Profile
using ThreadsX
using TimerOutputs
using Distances
#using ProfileView

# debug
const to = TimerOutput()


n = 100*1000 # has to be divisible by threads
vmax = 0.03
d0 = 0.02
dc = 0.01
l0 = 0.31
l1 = 0.001
l2 = 1.2
l3 = 2
l4 = 0.01

mutable struct Boid
    position::SVector{3, Float64}
    velocity::SVector{3, Float64}
end

@inline norm(a1) = sqrt((a1[1])^2 + (a1[2])^2 + (a1[3])^2)
@inline distance(a1, a2) = sqrt((a1[1] - a2[1])^2 + (a1[2] - a2[2])^2 + (a1[3] - a2[3])^2)
init_boid()::Boid = Boid(1.0 .- 2 * rand(Float64,3), 0.001 .* rand(Float64,3))

# update function
function update_boid(b, blist, nearby)
    # nearby has to be sorted
    n_length = 0
    c_length = 0

    s1 = zeros(SVector{3})
    s2 = zeros(SVector{3})
    s3 = zeros(SVector{3})
    @inbounds for i in 1:length(nearby)
        nb = blist[nearby[i]]
        d = distance(nb.position, b.position)
        if d < d0
            s1 += nb.position
            s2 += nb.velocity
            if d < dc
                s3 += nb.position - b.position
                c_length += 1
            end
            n_length += 1
        else
            # we know that the list is sorted so, when the first element is not in sight the others are even further away
            break
        end
    end

    w1 = (1 / n_length * s1) - b.position
    w2 = (1 / n_length * s2)
    w3 = (-1 / c_length * s3)
    w4 = maximum(abs.(b.position)) <= 1 ? zeros(3) : -b.position

    velocity = b.velocity * l0 + w1 * l1 + w2 * l2 + w3 * l3 + w4 * l4

    vnorm = norm(velocity)
    if vnorm > vmax
        velocity = vmax * (velocity / vnorm)
    end

    position = b.position + velocity
    Boid(position,velocity)
end


locations(boids) = reduce(hcat, [x.position for x in boids])

# init
boids = [init_boid() for _ in 1:n]
kdtree = KDTree(locations(boids); leafsize = 10)

fig = Figure()
ax3d = Axis3(fig[1, 1]; aspect = (1, 1, 1),
    perspectiveness = 0.3, azimuth = 0.8, elevation = 0.30, show_axis = false)
#hidedecorations!(ax3d)
#hidespines!(ax3d)
points = Observable([Point3f(x.position[1], x.position[2], x.position[3]) for x in boids])
#velocities = Observable([Point3f(x.velocity[1], x.velocity[2], x.velocity[3]) for x in boids])
#arrows!(ax3d,points,velocities,linewidth = 0.01, arrowsize = Vec3f(0.2, 0.2, 0.3)*0.1, linecolor=:gray60, arrowcolor=:gray30)
scatter!(ax3d,points,markersize=10)
display(fig)



# multithreading
threads = 8

# pre alloc buffer
buffer = [Vector{Boid}(undef,n÷threads) for t in 1:threads]
for t in 1:threads
    @inbounds for i in eachindex(buffer[t])
        buffer[t][i] = Boid(zeros(SVector{3}), zeros(SVector{3}))
    end
end


function update_sim() 
    @timeit to "begin" begin
    positions = [(i, x.position) for (i, x) in enumerate(boids)]
    shape = reshape(positions, (threads, div(length(positions), threads)))
    end

    @timeit to "locating" begin
    nearby = ThreadsX.map((i) -> hcat([x[1] for x in shape[i,:]] ,first(knn(kdtree, [x[2] for x in shape[i,:]], 10, true))),1:threads)
    end

    @timeit to "process" begin
        Threads.@threads for t = 1:threads
            nb = nearby[t]
            thread_buffer = buffer[t]
            @inbounds for i = 1:size(nb)[1]
                index = nb[i, 1]
                data = nb[i, 2]
                thread_buffer[i] = update_boid(boids[index], boids, data)
            end
        end
        global boids = reduce(vcat, buffer)
    end

    @timeit to "cleanup" begin
    # updated indices
    global kdtree = KDTree(locations(boids); leafsize = 10, reorder = true)
    end
end

(for t in 1:200
    println(t)
    @timeit to "frame" update_sim()
    points[] = [Point3f(x.position[1], x.position[2], x.position[3]) for x in boids]
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