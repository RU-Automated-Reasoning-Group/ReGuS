# StructuredProgramSearch
Search Karel Program (for now)





## Questions 

Apr. 17

- Hi professor, sorry for disrupt you again. I think I may misunderstood the meaning of "go back to the main branch", or "go back to anything"

- Suppose in the TopOff environment, the init_pos has NO marker, let's use **[NO]** to denote. We only need to record **[NO]** and **[YES]**, they denote [there is (a) marker(s)] and [there is no marker], respectively.

- (I am talking about the situation 2)

  

1. In the first prediction, we should predict MOVE, and then restart immediately.

  ```
  Program:
  {
   [NO - a] move
  }
  ```

​		We found that after a few iterations, the current_abs_state is [YES - b], so I cannot execute [NO - a] move directly, we need a new branch

```
Program:
{
    if ([YES]) {
        [YES - b] (action)
    }
    [NO - a] move
}
```



2. In the second iteration, let's fill this program. Suppose the prediction is put_marker, and now the program is

  ```
  Program:
  {
   if ([YES]) {
       [YES - b] put_marker
       [YES - c] (action)
   }
   [NO - a] move
  }
  
  ```

  Well, according to our algorithm, we should let this branch to go back to [NO - a] and terminate. We know that [YES - c] can never be Identical to [NO - a], so we may need to compare the **POST_ABS_STATE**? But this may give us a wrong program?


3. In the third iteration, let's fill this program and try to terminate this branch.
  According to our algorithm, the prediction of (action) should be identical to the action in the main branch (that we want to go back to).

  ```
  Program:
  {
   if ([YES]) {
       [YES - b] put_marker
       [YES - c] (move)     [? - d]
   }
   [NO - a] move
  }
  ```

  Now let's discuss the termination. We know that after executing the put_marker, the abs_state is definitely [YES - c],
  And we need to hope [? - d] is [NO - d] so that we can satisfy [NO - a].

There are at least two problems here:
- The [? - d] doesn't guaranteed to be [NO - d]. This is simple, because two consecutive markers can present.
  Even we do put_marker and then move, in the next position, a marker can present there, so we cannot execute [NO - a] move.

- Okay, let's make it simple, suppose in the next position, there is no marker. Then the program should be

```
Program:
{
 if ([YES]) {
     [YES - b] put_marker

 REMOVED([YES - c](move)[NO - d])

 }
 [NO - a] move
}
```

​	However, when we go back to [NO - a] from [NO - d], this doesn't make so much sense! We EXECUTE the (move) and get [NO - d], as if we execute move in the main branch (we should skip the [NO - a] move for a single time since we have already executed removed move]). The intuition is that we somehow fix the abs_state in the sub_branch, so that we can execute the code in the main branch. But our program is wrong here.



- A better way to explain what I am talking about is to USE the program:

  ```Program:
  {
      if ([YES]) {
          [YES - b] put_marker
      }
      [NO - a] move
  }

- This is wrong! The real GT program is: 

  ```
  Program:
    {
        if ([YES]) {
            [YES - b] put_marker
        }
        [YES/NO - a] move
    }
  
  ```

  Do you see the difference?



- A potential solution can be use "multiple" program for different iterations, for example, to solve TopOff task, we need two programs 

- ```
  Program:
  {
      move
  }
  ```

  

- ```
  Program:
  {
      put_marker
      move
  }
  ```

  

- I don't put any if there, just like DRL method. May be we can examine and summarize and then combines them? I will think about this tonight.

