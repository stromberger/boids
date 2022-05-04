using Random
using GLMakie
using Base.Threads
using NearestNeighbors
using StaticArrays
using Profile
using ThreadsX
#using ProfileView

n = 100*500
vmax = 0.03
d0 = 0.02
dc = 0.01
l0 = 0.31
l1 = 0.001
l2 = 1.2
l3 = 2
l4 = 0.01

struct Boid
    position::SVector{3, Float64}
    velocity::SVector{3, Float64}
end

norm(a1) = sqrt((a1[1])^2 + (a1[2])^2 + (a1[3])^2)
distance(a1, a2) = sqrt((a1[1] - a2[1])^2 + (a1[2] - a2[2])^2 + (a1[3] - a2[3])^2)
init_boid()::Boid = Boid(1.0 .- 2 * rand(Float64,3), 0.001 .* rand(Float64,3))

# update function
function update_boid(b, blist, nearby)
    # nearby has to be sorted
    n_length = 0
    c_length = 0

    s1 = zeros(SVector{3})
    s2 = zeros(SVector{3})
    s3 = zeros(SVector{3})
    for i in 1:length(nearby)
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
    Boid(position, velocity)
end


locations(boids::Vector{Boid}) = [x.position for x in boids]

# init
boids = [init_boid() for _ in 1:n]
kdtree = KDTree(locations(boids); leafsize = 10)

fig = Figure()
ax3d = Axis3(fig[1, 1]; aspect = (1, 1, 1),
    perspectiveness = 0.3, azimuth = 0.8, elevation = 0.30, show_axis = false)
#hidedecorations!(ax3d)
#hidespines!(ax3d)
points = Observable([Point3f(x.position[1], x.position[2], x.position[3]) for x in boids])
scatter!(ax3d,points,markersize=10)
display(fig)


buffer = copy(boids)

function update_sim() 
    threads = 8
    positions = [(i, x.position) for (i, x) in enumerate(boids)]
    shape = reshape(positions, (threads, div(length(positions), threads)))

    nearby = reduce(vcat, ThreadsX.map((i) -> hcat([x[1] for x in shape[i,:]] ,first(knn(kdtree, [x[2] for x in shape[i,:]], 25, true))),1:threads))

    #nearby2 =  first(knn(kdtree, positions, 25, true))

    Threads.@threads for i = 1:length(buffer)
        index = nearby[i, 1]
        data = nearby[i, 2]
        buffer[index] = update_boid(boids[index], boids, data)
    end
    global boids = buffer
    # updated indices
    global kdtree = KDTree(locations(boids); leafsize = 10, reorder = true)
end

 (for t in 1:2000
    println(t)
    update_sim()
    points[] = [Point3f(x.position[1], x.position[2], x.position[3]) for x in boids]
    #sleep(0.0001)
end)



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